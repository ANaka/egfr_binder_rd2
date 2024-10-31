import modal
from typing import List
from pathlib import Path
import pandas as pd
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from egfr_binder_rd2 import (
    MODAL_VOLUME_NAME,
    OUTPUT_DIRS,
    LOGGING_CONFIG,
    MODAL_VOLUME_PATH
)
from egfr_binder_rd2.datamodule import SequenceDataModule
from egfr_binder_rd2.bt import BTRegressionModule

import logging

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
    )
)

app = modal.App("bt-training")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Create a secret for wandb
wandb_secret = modal.Secret.from_name("anaka_personal_wandb_api_key")

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={MODAL_VOLUME_PATH: volume},
    secrets=[wandb_secret],  # Add the secret here
)
def train_bt_model(
    yvar: str,
    wandb_project: str,
    wandb_entity: str,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 32,
    max_epochs: int = 40,
    learning_rate: float = 1e-3,
    peft_r: int = 8,
    peft_alpha: int = 16,
    max_length: int = 512,
    transform_type: str = "rank",
    make_negative: bool = False,
    seed: int = 42,
):
    """Train a Bradley-Terry regression model on sequence data."""
    # Set random seeds
    torch.manual_seed(seed)
    
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
    df = pd.read_csv(Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["rd1_fold_df"]).rename(columns={"binder_sequence": "Sequence"})

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
    
    # Initialize model
    model = BTRegressionModule(
        label=yvar,
        model_name=model_name,
        lr=learning_rate,
        peft_r=peft_r,
        peft_alpha=peft_alpha,
        max_length=max_length,
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
    
    # Save adapter
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["bt_models"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"bt_{yvar}_{transform_type}.pt"
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

@app.local_entrypoint()
def main():
    # Example usage
    train_bt_model.remote(
        yvar="pae_interaction",
        wandb_project="bt_regression",
        wandb_entity="anaka_personal",
        transform_type="rank",
        make_negative=True,
    )
