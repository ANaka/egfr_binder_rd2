import modal
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, EsmForMaskedLM
import re
from datetime import datetime
from dataclasses import dataclass
import enum
from typing import Any

import evo_prot_grad

from egfr_binder_rd2 import (
    MODAL_VOLUME_NAME,
    OUTPUT_DIRS,
    LOGGING_CONFIG,
    MODAL_VOLUME_PATH,
    ExpertType,
    ExpertConfig,
)
from egfr_binder_rd2.datamodule import SequenceDataModule
from egfr_binder_rd2.bt import BTRegressionModule
from egfr_binder_rd2.esm_regression_expert import EsmRegressionExpert
import logging
from egfr_binder_rd2.utils import hash_seq


# Set up logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# Define container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "lightning",
        "wandb",
        "datasets",
        "peft",
        "torchmetrics",
        "evo-prot-grad",
    )
)

app = modal.App("bt-training")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Create a secret for wandb
wandb_secret = modal.Secret.from_name("anaka_personal_wandb_api_key")



def create_expert(config: ExpertConfig, device: str) -> Any:
    """Create an expert based on configuration."""
    if config.type == ExpertType.ESM:
        return evo_prot_grad.get_expert(
            expert_name='esm',
            scoring_strategy="mutant_marginal",
            model=EsmForMaskedLM.from_pretrained(config.model_name),
            tokenizer=AutoTokenizer.from_pretrained(config.model_name),
            device=device,
            temperature=config.temperature,
        )
    else:
        # Find latest adapter for this expert type
        model_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["bt_models"]
        model_path = get_latest_adapter(
            model_dir, 
            config.type.value, 
            config.transform_type
        )
        
        bt_model = BTRegressionModule.load_adapter(model_path)
        bt_model.eval()
        
        return EsmRegressionExpert(
            temperature=config.temperature,
            model=bt_model,
            tokenizer=bt_model.tokenizer,
            device=device,
        )

def get_latest_adapter(base_path: Path, yvar: str, transform_type: str) -> Path:
    """Find the most recent adapter file for given yvar and transform type.
    
    The expected format is: bt_{yvar}_{transform_type}_{timestamp}.pt
    """
    pattern = f"bt_{yvar}_{transform_type}_*.pt"
    adapter_files = list(base_path.glob(pattern))
    
    if not adapter_files:
        # Fall back to the old naming format if no timestamped files exist
        old_pattern = f"bt_{yvar}_{transform_type}.pt"
        old_file = base_path / old_pattern
        if old_file.exists():
            return old_file
        raise FileNotFoundError(f"No adapter files found matching pattern: {pattern}")
    
    # Sort by timestamp in filename
    adapter_files.sort(key=lambda x: x.stem.split('_')[-1], reverse=True)
    return adapter_files[0]

def create_experts(expert_configs: Optional[List[ExpertConfig]], device: str) -> List[Any]:
    """Create multiple experts based on configurations.
    
    Args:
        expert_configs: List of expert configurations
        device: Device to run the experts on ('cuda' or 'cpu')
        
    Returns:
        List of initialized experts
    """
    # Use default configuration if none provided
    if expert_configs is None:
        expert_configs = [
            ExpertConfig(
                type=ExpertType.ESM,
                weight=1.0,
                temperature=1.0,
            ),
            ExpertConfig(
                type=ExpertType.iPAE,
                weight=1.0,
                temperature=1.0,
                make_negative=True,
                transform_type="rank",
            ),
        ]
    
    # Create experts
    experts = []
    for config in expert_configs:
        try:
            expert = create_expert(config, device)
            if config.weight != 1.0:
                expert = evo_prot_grad.WeightedExpert(expert, config.weight)
            experts.append(expert)
        except Exception as e:
            logger.warning(f"Failed to create expert {config.type}: {e}")
            continue
    
    if not experts:
        raise ValueError("No experts could be created!")
    
    return experts

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={MODAL_VOLUME_PATH: volume},
    secrets=[wandb_secret],  # Add the secret here
)
def train_bt_model(
    yvar: str,
    wandb_project: str,
    wandb_entity: str,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 48,
    max_epochs: int = 40,
    learning_rate: float = 1e-3,
    peft_r: int = 8,
    peft_alpha: int = 16,
    max_length: int = 512,
    transform_type: str = "rank",
    make_negative: bool = False,
    seed: int = 117,
    xvar: str = 'binder_sequence',  # Add xvar parameter
):
    """Train a Bradley-Terry regression model on sequence data."""
    # Validate yvar is a supported type
    try:
        expert_type = ExpertType.from_str(yvar)
    except ValueError:
        raise ValueError(f"Unsupported yvar: {yvar}. Must be one of {[e.value for e in ExpertType]}")
    
    # Set random seeds
    torch.manual_seed(seed)
    
    # Set matmul precision
    torch.set_float32_matmul_precision('high')
    
    # Initialize wandb run - no need to call wandb.login() as the API key is set via environment variable
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "yvar": yvar,
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "peft_r": peft_r,
            "peft_alpha": peft_alpha,
            "max_length": max_length,
            "transform_type": transform_type,
            "make_negative": make_negative,
            "seed": seed,
            "max_epochs": max_epochs,
        }
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    old_df = pd.read_csv(Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["rd1_fold_df"])
    df = pd.read_csv(Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["metrics_csv"])
    df = df.merge(old_df, how='outer')

    # TODO add in stuff from newly folded sequences here
    
    # Create data module
    data_module = SequenceDataModule(
        df=df,
        tokenizer_name=model_name,
        yvar=yvar,
        batch_size=batch_size,
        max_length=max_length,
        transform_type=transform_type,
        make_negative=make_negative,
        seed=seed,
    )
    data_module.setup()
    
    # Initialize model with xvar
    model = BTRegressionModule(
        label=yvar,
        model_name=model_name,
        lr=learning_rate,
        peft_r=peft_r,
        peft_alpha=peft_alpha,
        max_length=max_length,
        xvar=xvar,  # Pass xvar to model
    )
    
    # Set up W&B logger with existing run
    wandb_logger = WandbLogger(experiment=run)
    
    # Set up callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_spearman",
        min_delta=0.00,
        patience=30,
        verbose=False,
        mode="max"
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop_callback],
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        val_check_interval=0.25,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Get predictions with sequences
    logger.info("Generating prediction tables...")
    train_preds = trainer.predict(model, data_module.train_dataloader(shuffle=False))
    val_preds = trainer.predict(model, data_module.val_dataloader())
    test_preds = trainer.predict(model, data_module.test_dataloader())

    # Convert predictions and sequences to lists
    train_df = pd.DataFrame({
        'predictions': torch.cat([x['predictions'] for x in train_preds]).cpu().numpy(),
        'sequence': [seq for batch in train_preds for seq in batch['sequence']],
    })
    val_df = pd.DataFrame({
        'predictions': torch.cat([x['predictions'] for x in val_preds]).cpu().numpy(),
        'sequence': [seq for batch in val_preds for seq in batch['sequence']],
    })
    test_df = pd.DataFrame({
        'predictions': torch.cat([x['predictions'] for x in test_preds]).cpu().numpy(),
        'sequence': [seq for batch in test_preds for seq in batch['sequence']],
    })

    # Add sequence info and other metrics
    train_df[yvar] = data_module.train_dataset[yvar].numpy()
    val_df[yvar] = data_module.val_dataset[yvar].numpy()
    test_df[yvar] = data_module.test_dataset[yvar].numpy()

    # Add additional metrics
    for df, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        df['pred_rank'] = df['predictions'].rank()
        df['true_rank'] = df[yvar].rank()
        df['residuals'] = df[yvar] - df['predictions']
        df['split'] = split_name
        df['length'] = df['sequence'].str.len()
        df['hash'] = df['sequence'].apply(hash_seq)

    # Log prediction tables to wandb
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        wandb_logger.experiment.log({
            f"{split_name}_predictions": wandb.Table(dataframe=df)
        })
    
    # Save adapter with timestamp
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["bt_models"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"bt_{yvar}_{transform_type}_{timestamp}.pt"
    model.save_adapter(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Log model to W&B
    artifact = wandb.Artifact(
        name=f"bt_{yvar}_{transform_type}_model",
        type="model",
        description=f"Bradley-Terry model trained on {yvar}"
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    
    # Close W&B
    wandb.finish()
    
    return str(model_path)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={MODAL_VOLUME_PATH: volume},
    secrets=[wandb_secret],
)
def sample_sequences(
    sequences: List[str],                   # List of starting sequences
    expert_configs: Optional[List[ExpertConfig]] = None,
    n_parallel_chains: int = 32,             # Number of parallel chains per sequence per run
    n_serial_chains: int = 1,               # Number of sequential runs per sequence
    n_steps: int = 250,                     # Steps per chain
    max_mutations: int = -1,                 # Max mutations per sequence
    seed: int = 42,                         # Random seed
) -> pd.DataFrame:
    """Sample sequences using EvoProtGrad with multiple experts and serial chains.
    
    Args:
        sequences: List of starting sequences
        expert_configs: List of expert configurations
        n_parallel_chains: Number of parallel chains per sequence per run
        n_serial_chains: Number of sequential runs per sequence
        n_steps: Number of steps per chain
        max_mutations: Maximum mutations allowed per sequence
        seed: Random seed
    """
    # Set random seeds
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    
    # Initialize experts (same as before)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = create_experts(expert_configs, device)
    
    # Initialize results storage
    results = []
    total_chains = 0
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Process each starting sequence
    for seq_idx, sequence in enumerate(sequences):
        # Get hash of parent sequence
        parent_hash = hash_seq(sequence)
        
        logger.info(f"Processing sequence {seq_idx + 1}/{len(sequences)}: {sequence} (hash: {parent_hash})")
        
        # Run serial chains for this sequence
        for serial_idx in range(n_serial_chains):
            logger.info(f"Running serial chain {serial_idx + 1}/{n_serial_chains} for sequence {seq_idx + 1}")
            
            variants, scores = evo_prot_grad.DirectedEvolution(
                wt_protein=sequence,
                output="all",
                experts=experts,
                parallel_chains=n_parallel_chains,
                n_steps=n_steps,
                max_mutations=max_mutations,
                verbose=False,
            )()
            
            # Process results
            for chain in range(scores.shape[1]):
                for step in range(scores.shape[0]):
                    seq = "".join(variants[step][chain].split(" "))
                    score = float(scores[step, chain])
                    results.append({
                        "run": timestamp,
                        "parent_idx": seq_idx,
                        "parent_seq": sequence,
                        "parent_hash": parent_hash,
                        "chain": total_chains,
                        "step": step,
                        "score": score,
                        "sequence": seq,
                        "sequence_hash": hash_seq(seq),
                        "length": len(seq),
                    })
                total_chains += 1
    
    # Convert to DataFrame, remove duplicates, and sort by score
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=['sequence_hash'])
    df = df.sort_values('score', ascending=False)
    
    logger.info(f"Generated {len(df)} unique sequences across {total_chains} chains")
    return df

@app.local_entrypoint()
def main():
    # Example usage with multiple experts
    expert_configs = [
        ExpertConfig(
            type=ExpertType.ESM,
            weight=1.0,
            temperature=1.0,
        ),
        ExpertConfig(
            type=ExpertType.iPAE,
            weight=1.0,
            temperature=1.0,
            make_negative=True,
        ),
    ]
    
    sequences = sample_sequences.remote(
        sequence="AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS",
        expert_configs=expert_configs,
        n_parallel_chains=32,           # Run 32 parallel chains
        n_serial_chains=8,             # Run 8 times sequentially
        n_steps=50,                    # 50 steps per chain
        max_mutations=4,               # Max 4 mutations per sequence
    )
    print(f"Generated {len(sequences)} sequences")

@app.local_entrypoint()
def train():
    train_bt_model.remote(
        yvar="pae_interaction",
        wandb_project="egfr-binder-rd2", 
        wandb_entity="anaka_personal",
        model_name="facebook/esm2_t6_8M_UR50D",
        batch_size=32,
        max_epochs=10,
        learning_rate=1e-3,
        peft_r=8,
        peft_alpha=16,
        max_length=512,
        transform_type="rank",
        make_negative=False,
        seed=117,
    )

