import modal
from typing import List, Optional, Dict, Tuple
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
import numpy as np
from sklearn.model_selection import KFold
import time
from functools import wraps
import random

import evo_prot_grad

from egfr_binder_rd2 import (
    MODAL_VOLUME_NAME,
    OUTPUT_DIRS,
    LOGGING_CONFIG,
    MODAL_VOLUME_PATH,
    ExpertType,
    ExpertConfig,
    PartialEnsembleExpertConfig,
)
from egfr_binder_rd2.datamodule import SequenceDataModule
from egfr_binder_rd2.bt import BTRegressionModule, PartialEnsembleModule
from egfr_binder_rd2.esm_regression_expert import EsmRegressionExpert
from egfr_binder_rd2.esm2_pll import get_esm2_pll
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
        "scikit-learn",
    )
)

app = modal.App("bt-training")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Create a secret for wandb
wandb_secret = modal.Secret.from_name("anaka_personal_wandb_api_key")



def create_expert(config: ExpertConfig, device: str) -> Any:
    """Create an expert based on configuration."""
    logger.info(f"Creating expert with config: {config}")
    
    if config.type == ExpertType.ESM:
        logger.info(f"Loading ESM model from {config.model_name}")
        model = EsmForMaskedLM.from_pretrained(config.model_name)
        # Ensure model is in eval mode
        model.eval()
        logger.info("Creating ESM expert with mutant_marginal scoring strategy")
        return evo_prot_grad.get_expert(
            expert_name='esm',
            scoring_strategy="mutant_marginal", 
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(config.model_name),
            device=device,
            temperature=config.temperature,
        )
    else:
        try:
            # Find latest adapter for this expert type
            model_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["bt_models"]
            logger.info(f"Looking for {config.type.value} adapter in {model_dir}")
            model_path = get_latest_adapter(
                model_dir, 
                config.type.value, 
                config.transform_type,
                use_ensemble=isinstance(config, PartialEnsembleExpertConfig)
            )
            logger.info(f"Found adapter at {model_path}")
            
            # Load appropriate model type
            if isinstance(config, PartialEnsembleExpertConfig):
                logger.info("Loading ensemble model")
                bt_model = PartialEnsembleModule.load_model(model_path)
            else:
                logger.info("Loading BT model")
                bt_model = BTRegressionModule.load_adapter(model_path)
            bt_model.eval()
            
            logger.info("Creating ESM regression expert")
            return EsmRegressionExpert(
                temperature=config.temperature,
                model=bt_model,
                tokenizer=bt_model.tokenizer,
                device=device,
            )
        except Exception as e:
            logger.error(f"Failed to create expert {config.type}: {str(e)}")
            raise

def get_latest_adapter(base_path: Path, yvar: str, transform_type: str, use_ensemble: bool = False) -> Path:
    """Find the most recent adapter file for given yvar and transform type."""
    model_type = "ensemble" if use_ensemble else "bt"
    pattern = f"{model_type}_{yvar}_{transform_type}_*.pt"
    adapter_files = list(base_path.glob(pattern))
    
    if not adapter_files:
        # Fall back to the old naming format if no timestamped files exist
        old_pattern = f"{model_type}_{yvar}_{transform_type}.pt"
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
                temperature=1.0,
            ),
            PartialEnsembleExpertConfig(
                type=ExpertType.iPAE,
                temperature=1.0,
                make_negative=True,
                transform_type="standardize",
            ),
            PartialEnsembleExpertConfig(
                type=ExpertType.iPTM,
                temperature=1.0,
                make_negative=False,
                transform_type="standardize",
            ),
            PartialEnsembleExpertConfig(
                type=ExpertType.pLDDT,
                temperature=1.0,
                make_negative=False,
                transform_type="standardize",
            ),
        ]
    
    # Create experts
    experts = []
    for config in expert_configs:
        try:
            expert = create_expert(config, device)
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
    num_heads: int = 5,  # New parameter for PartialEnsembleModule
    dropout: float = 0.1,  # New parameter for PartialEnsembleModule
    explore_weight: float = 0.2,  # New parameter for PartialEnsembleModule
    use_ensemble: bool = True,  # Flag to choose between BTRegressionModule and PartialEnsembleModule
):
    """Train a Bradley-Terry regression model or ensemble on sequence data."""
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
    
    # Initialize model based on use_ensemble flag
    if use_ensemble:
        model = PartialEnsembleModule(
            label=yvar,
            num_heads=num_heads,
            model_name=model_name,
            lr=learning_rate,
            peft_r=peft_r,
            peft_alpha=peft_alpha,
            max_length=max_length,
            xvar=xvar,
            dropout=dropout,
            explore_weight=explore_weight,
        )
    else:
        model = BTRegressionModule(
            label=yvar,
            model_name=model_name,
            lr=learning_rate,
            peft_r=peft_r,
            peft_alpha=peft_alpha,
            max_length=max_length,
            xvar=xvar,
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
    def create_pred_df(predictions_list, name=""):
        try:
            # Validate predictions
            if not predictions_list or any(x is None for x in predictions_list):
                raise ValueError(f"Empty or None predictions in {name} set")
                
            # Validate prediction tensors and sequences
            pred_tensors = []
            sequences = []
            for batch in predictions_list:
                if 'predictions' not in batch or 'sequence' not in batch:
                    raise ValueError(f"Missing predictions or sequence in {name} batch")
                    
                pred_tensor = batch['predictions'].view(-1, 1)
                if not torch.is_tensor(pred_tensor) or pred_tensor.numel() == 0:
                    raise ValueError(f"Invalid prediction tensor in {name} batch")
                    
                pred_tensors.append(pred_tensor)
                sequences.extend(batch['sequence'])
            
            # Validate sequences are strings
            if not all(isinstance(seq, str) for seq in sequences):
                raise ValueError(f"Non-string sequences found in {name} set")
                
            # Create dataframe
            df = pd.DataFrame({
                'predictions': torch.cat(pred_tensors).cpu().numpy().squeeze(),
                'sequence': sequences,
            })
            
            # Add hash column only if sequences are valid
            df['hash'] = df['sequence'].apply(lambda x: hash_seq(x) if isinstance(x, str) else None)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating {name} predictions dataframe: {str(e)}")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['predictions', 'sequence', 'hash'])

    # Convert predictions and sequences to dataframes
    train_df = create_pred_df(train_preds, "train") 
    val_df = create_pred_df(val_preds, "validation")
    test_df = create_pred_df(test_preds, "test")

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
    
    # Save model based on type
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["bt_models"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "ensemble" if use_ensemble else "bt"
    model_path = output_dir / f"{model_type}_{yvar}_{transform_type}_{timestamp}.pt"
    
    if use_ensemble:
        model.save_model(model_path)
    else:
        model.save_adapter(model_path)
    
    logger.info(f"Saved model to {model_path}")
    
    # Log model to W&B
    artifact = wandb.Artifact(
        name=f"{model_type}_{yvar}_{transform_type}_model",
        type="model",
        description=f"{'Ensemble' if use_ensemble else 'Bradley-Terry'} model trained on {yvar}"
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    
    volume.commit()
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
    sequences: List[str],
    expert_configs: Optional[List[ExpertConfig]] = None,
    n_parallel_chains: int = 32,
    n_serial_chains: int = 1,
    n_steps: int = 250,
    max_mutations: int = -1,
    seed: int = 42,
    run_inference: bool = True,  # New parameter
) -> pd.DataFrame:
    """Sample sequences using EvoProtGrad with multiple experts and serial chains.
    
    Args:
        sequences: List of sequences to start from
        expert_configs: List of expert configurations
        n_parallel_chains: Number of parallel chains to run
        n_serial_chains: Number of times to run chains sequentially
        n_steps: Number of steps per chain
        max_mutations: Maximum mutations per sequence (-1 for no limit)
        seed: Random seed
        run_inference: Whether to run detailed inference with non-ESM experts
    """
    # Set random seeds
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    
    # Initialize experts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = create_experts(expert_configs, device)
    
    logger.info(f"Using experts: {[e for e in experts]}")

    # Initialize results storage
    results = []
    total_chains = 0
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Process each starting sequence
    for seq_idx, sequence in enumerate(sequences):
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
    
    # Convert to initial DataFrame and remove duplicates
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=['sequence_hash']).reset_index(drop=True)
    
    if run_inference:
        logger.info("Getting detailed predictions from each expert...")
        for expert, config in zip(experts, expert_configs):
            if config.type == ExpertType.ESM:
                logger.info(f"Getting ESM2 PLL scores for {config} expert")
                # Get ESM2 PLL scores using the already loaded model
                sequences_list = df['sequence'].tolist()
                batch_size = 32
                
                all_results = get_esm2_pll(
                    model=expert.model,  # The ESM model is already loaded in the expert
                    tokenizer=expert.tokenizer,
                    sequences=sequences_list,
                    batch_size=batch_size
                )
                
                # Extract sequence_log_pll from results and add to DataFrame
                _df = pd.DataFrame(all_results)
                df['sequence_log_pll'] = _df['sequence_log_pll']
            else:
                logger.info(f"Getting detailed predictions for {config} expert")
                # Process non-ESM experts
                model = expert.model
                sequences = df['sequence'].tolist()
                expert_prefix = f"{config.type.value}"
                
                # Process in batches to avoid memory issues
                batch_size = 32
                predictions = []
                uncertainties = []
                ucbs = []
                head_predictions = []
                
                for i in range(0, len(sequences), batch_size):
                    batch_seqs = sequences[i:i + batch_size]
                    batch = model.tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
                    
                    with torch.no_grad():
                        if isinstance(model, PartialEnsembleModule):
                            outputs = model(batch)
                            # Flatten the predictions using ravel()
                            predictions.extend(outputs['predictions'].cpu().numpy().ravel())
                            uncertainties.extend(outputs['uncertainties'].cpu().numpy().ravel())
                            ucbs.extend(outputs['ucb'].cpu().numpy().ravel())
                            head_predictions.append(outputs['head_predictions'].cpu().numpy())
                        else:  # Regular BT model
                            outputs = model(batch)
                            predictions.extend(outputs['predictions'].cpu().numpy().ravel())
                
                # Add results to DataFrame
                if isinstance(model, PartialEnsembleModule):
                    df[f"{expert_prefix}_mean"] = predictions
                    df[f"{expert_prefix}_std"] = uncertainties
                    df[f"{expert_prefix}_ucb"] = ucbs
                    head_predictions = np.concatenate(head_predictions, axis=0)
                    for head_idx in range(head_predictions.shape[1]):
                        df[f"{expert_prefix}_head_{head_idx}"] = head_predictions[:, head_idx]
                else:
                    df[f"{expert_prefix}_pred"] = predictions
    
    # Sort by score
    df = df.sort_values('score', ascending=False)
    
    logger.info(f"Generated {len(df)} unique sequences across {total_chains} chains")
    return df

@app.local_entrypoint()
def main():
    # Example usage with multiple experts
    expert_configs = [
        ExpertConfig(
                type=ExpertType.ESM, 
                temperature=1.0,
            ),
        PartialEnsembleExpertConfig(
            type=ExpertType.iPAE,
            temperature=1.0,
            make_negative=True,
            transform_type="standardize",
        ),
        PartialEnsembleExpertConfig(
            type=ExpertType.iPTM,
            temperature=1.0,
            make_negative=False,
            transform_type="standardize",
        ),
        PartialEnsembleExpertConfig(
            type=ExpertType.pLDDT,
            temperature=1.0,
            make_negative=False,
            transform_type="standardize",
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
        max_epochs=30,
        learning_rate=1e-3,
        peft_r=8,
        peft_alpha=16,
        max_length=512,
        transform_type="standardize",
        make_negative=True,
        seed=117,
        use_ensemble=True,
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    )
    train_bt_model.remote(
        yvar="i_ptm",
        wandb_project="egfr-binder-rd2", 
        wandb_entity="anaka_personal",
        model_name="facebook/esm2_t6_8M_UR50D",
        batch_size=32,
        max_epochs=30,
        learning_rate=1e-3,
        peft_r=8,
        peft_alpha=16,
        max_length=512,
        transform_type="standardize",
        make_negative=False,
        seed=117,
        use_ensemble=True,
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    )
    train_bt_model.remote(
        yvar="binder_plddt",
        wandb_project="egfr-binder-rd2", 
        wandb_entity="anaka_personal",
        batch_size=32,
        max_epochs=30,
        learning_rate=1e-3,
        peft_r=8,
        peft_alpha=16,
        max_length=512,
        transform_type="standardize",
        make_negative=False,
        seed=117,
        use_ensemble=True,
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    )


