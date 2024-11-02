import torch
import lightning as pl
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, SpearmanCorrCoef
from transformers import EsmForSequenceClassification, AutoTokenizer, EsmModel
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict, get_peft_model_state_dict
import wandb
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import hashlib
import pandas as pd
from egfr_binder_rd2.utils import hash_seq
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
# from esm.modules import TransformerLayer
# Utility functions
def create_pairwise_comparisons(batch, outputs, label):
    device = outputs.device
    n_samples = outputs.shape[0]

    if n_samples < 2:
        return None, None, None

    pairs = torch.combinations(torch.arange(n_samples), r=2)
    labels = batch[label].to(device)

    # Ensure outputs maintain gradients
    scores_i = outputs[pairs[:, 0]].clone()
    scores_j = outputs[pairs[:, 1]].clone()

    labels_i = labels[pairs[:, 0]]
    labels_j = labels[pairs[:, 1]]

    y_ij = (labels_i > labels_j).float()

    return scores_i, scores_j, y_ij

class BradleyTerryLoss(nn.Module):
    def forward(self, scores_i, scores_j, y_ij):
        prob_i_preferred = torch.sigmoid(scores_i - scores_j)
        return nn.functional.binary_cross_entropy(prob_i_preferred, y_ij)

class BTRegressionModule(pl.LightningModule):
    def __init__(
        self,
        label: str,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        lr: float = 5e-4,
        peft_r: int = 8,
        peft_alpha: int = 16,
        max_length: int = 512,
        xvar: str = 'binder_sequence',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.label = label
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm_model = self._initialize_model(model_name, peft_r, peft_alpha)

        self.bt_loss = BradleyTerryLoss()

        self.train_metrics = self._setup_train_metrics()
        self.val_mae = MeanAbsoluteError()
        self.val_spearman = SpearmanCorrCoef()

        self.loaded_adapters = {}
        self.adapter_run_paths = {}
        self.current_adapter_name = None

    def _initialize_model(self, model_name, peft_r, peft_alpha):
        base_model = EsmForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        peft_config = LoraConfig(
            r=peft_r, 
            lora_alpha=peft_alpha, 
            task_type=TaskType.SEQ_CLS,
            target_modules = ["query", "key", "value", "dense_h_to_4h", "dense_4h_to_h"],
        )
        return get_peft_model(base_model, peft_config)

    def _setup_train_metrics(self):
        return MetricCollection({
            "train_mae": MeanAbsoluteError(),
            "train_spearman": SpearmanCorrCoef(),
        })

    @property
    def adapter_names(self):
        return list(self.loaded_adapters.keys())

    def switch_adapter(self, index):
        if index < 0 or index >= len(self.loaded_adapters):
            raise ValueError(f"Invalid adapter index: {index}")

        adapter_name = self.adapter_names[index]
        if adapter_name != self.current_adapter_name:
            set_peft_model_state_dict(self.esm_model, self.loaded_adapters[adapter_name])
            self.current_adapter_name = adapter_name
            print(f"Switched to adapter: {adapter_name}")

    def forward(self, batch):
        input_ids = batch["input_ids"].squeeze(1) if batch["input_ids"].dim() > 2 else batch["input_ids"]
        attention_mask = batch["attention_mask"].squeeze(1) if batch["attention_mask"].dim() > 2 else batch["attention_mask"]

        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        batch["predictions"] = outputs.logits
        return batch

    def training_step(self, batch, batch_idx):
        batch = self(batch)
        outputs = batch["predictions"]

        loss = self._compute_bt_loss(batch, outputs)
        self.log("train_loss", loss)

        self._update_train_metrics(outputs, batch[self.label])

        return loss

    def _compute_bt_loss(self, batch, outputs):
        scores_i, scores_j, y_ij = create_pairwise_comparisons(
            batch, outputs.squeeze(), self.label
        )
        if scores_i is not None:
            bt_loss = self.bt_loss(scores_i, scores_j, y_ij)
            return bt_loss
        return torch.tensor(0.0, device=self.device)

    def _update_train_metrics(self, outputs, labels):
        pred = outputs.view(-1)
        target = labels
        if len(pred) > 0 and len(target) > 0:
            self.train_metrics(pred, target)

    def validation_step(self, batch, batch_idx):
        batch = self(batch)
        outputs = batch["predictions"]

        loss = self._compute_bt_loss(batch, outputs)
        self.log("val_loss", loss, sync_dist=True)

        self._update_validation_metrics(outputs, batch[self.label])

        return loss

    def _update_validation_metrics(self, outputs, labels):
        pred = outputs.view(-1)
        target = labels

        self.val_mae.update(pred, target)
        self.val_spearman.update(pred, target)

        current_val_spearman = self.val_spearman.compute()
        self.log("val_spearman", current_val_spearman, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        return {
            'predictions': outputs['predictions'].squeeze(),
            'sequence': batch[self.hparams.xvar],
        }

    @staticmethod
    def _get_sequence_info(sequence):
        """Calculate sequence length and hash."""
        return {
            'length': len(sequence),
            'hash': hash_seq(sequence)
        }

    def save_adapter(self, save_path):
        adapter_state_dict = get_peft_model_state_dict(self.esm_model)
        save_dict = {
            'adapter_state_dict': adapter_state_dict,
            'hyperparameters': self.hparams
        }
        torch.save(save_dict, save_path)
        print(f"Adapter and hyperparameters saved to: {save_path}")

    @classmethod
    def load_adapter(cls, load_path):
        saved_dict = torch.load(load_path)
        hparams = saved_dict['hyperparameters']

        model = cls(**hparams)
        set_peft_model_state_dict(model.esm_model, saved_dict['adapter_state_dict'])

        print(f"Adapter and hyperparameters loaded from: {load_path}")
        return model

    def tokenize_sequences(self, sequences):
        return self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def predict_sequences(self, sequences, batch_size=32):
        self.eval()
        predictions = []

        # Create a DataLoader if sequences is a Dataset, otherwise use it directly
        if isinstance(sequences, Dataset):
            data_loader = DataLoader(sequences, batch_size=batch_size)
        else:
            data_loader = [sequences[i:i+batch_size] for i in range(0, len(sequences), batch_size)]

        with torch.no_grad():
            for batch in tqdm(data_loader):
                inputs = self.tokenize_sequences(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Create a batch dictionary with the required format
                batch_dict = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                outputs = self(batch_dict)
                batch_predictions = outputs["predictions"].squeeze().tolist()
                predictions.extend(batch_predictions if isinstance(batch_predictions, list) else [batch_predictions])

        return predictions

    @classmethod
    def load_from_wandb(cls, run_path: str, model_artifact_name: str, run = None):
        """
        Load a model from a wandb artifact using the API or an existing run.

        Args:
            run_path (str): Path to the wandb run (e.g., 'username/project/run_id')
            model_artifact_name (str): Name of the model artifact
            run (wandb.sdk.wandb_run.Run, optional): An existing wandb run object. If None, a new API client will be used.

        Returns:
            BTRegressionModule: Loaded model
        """
        if run is None:
            api = wandb.Api()
            artifact = api.artifact(f'{run_path}/{model_artifact_name}:latest')
            artifact_dir = artifact.download()
        else:
            artifact = run.use_artifact(f'{model_artifact_name}:latest')
            artifact_dir = artifact.download()

        # Get the path to the downloaded model file
        model_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

        # Load the model
        model = cls.load_adapter(model_path)

        # Store the full run path
        model.current_run_path = f'{artifact.entity}/{artifact.project}/{artifact.source_name}'

        return model

@dataclass
class EnsembleMember:
    model: BTRegressionModule
    validation_score: float
    seed: int

class BTEnsemble:
    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.members: List[EnsembleMember] = []
        
    def add_member(self, model: BTRegressionModule, validation_score: float, seed: int):
        self.members.append(EnsembleMember(model, validation_score, seed))
        
    def predict(self, sequences: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Get predictions and uncertainties using ensemble"""
        all_predictions = []
        
        # Get predictions from each model
        for member in self.members:
            preds = member.model.predict_sequences(sequences, batch_size)
            all_predictions.append(preds)
            
        # Convert to numpy array for calculations
        predictions = np.array(all_predictions)
        
        # Calculate mean and std across ensemble members
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            "mean": mean_pred,
            "std": std_pred,
            "raw_predictions": predictions
        }

    def save(self, save_dir: str):
        """Save all ensemble members"""
        for i, member in enumerate(self.members):
            save_path = f"{save_dir}/ensemble_member_{i}.pt"
            member.model.save_adapter(save_path)
            
    @classmethod
    def load(cls, load_dir: str, n_models: int) -> 'BTEnsemble':
        """Load ensemble from directory"""
        ensemble = cls(n_models=n_models)
        for i in range(n_models):
            load_path = f"{load_dir}/ensemble_member_{i}.pt"
            model = BTRegressionModule.load_adapter(load_path)
            # Note: We'll need to recompute validation scores if needed
            ensemble.add_member(model, 0.0, i)  
        return ensemble

class PartialEnsembleHead(nn.Module):
    def __init__(
        self, 
        hidden_size,
        dropout=0.1
    ):
        super().__init__()
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using Kaiming initialization for ReLU networks"""
        for module in self.regression_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_in', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, pooler_output):
        return self.regression_head(pooler_output)

class PartialEnsembleModule(pl.LightningModule):
    def __init__(
        self,
        label: str,  # Changed from labels to label since all heads predict same target
        num_heads: int = 5,  # Number of ensemble heads
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        lr: float = 5e-4,
        peft_r: int = 8,
        peft_alpha: int = 16,
        max_length: int = 512,
        xvar: str = 'binder_sequence',
        dropout: float = 0.1,
        explore_weight: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        
        
        self.label = label
        self.max_length = max_length
        self.explore_weight = explore_weight
        # Initialize base components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm_model = self._initialize_model(model_name, peft_r, peft_alpha)
        
        # Create multiple independent heads predicting the same target
        config = self.esm_model.config
        self.ensemble_heads = nn.ModuleList([
            PartialEnsembleHead(
                hidden_size=config.hidden_size,
                dropout=dropout
            ) for _ in range(num_heads)
        ])
        
        # Initialize metrics
        self.train_metrics = MetricCollection({
            "train_mae": MeanAbsoluteError(),
            "train_spearman": SpearmanCorrCoef(),
        })
        self.val_mae = MeanAbsoluteError()
        self.val_spearman = SpearmanCorrCoef()

        # Add Bradley-Terry loss
        self.bt_loss = BradleyTerryLoss()

    def _initialize_model(self, model_name, peft_r, peft_alpha):
        base_model = EsmModel.from_pretrained(
            model_name,
        )
        peft_config = LoraConfig(
            r=peft_r, 
            lora_alpha=peft_alpha, 
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["query", "key", "value", "dense_h_to_4h", "dense_4h_to_h"],
        )
        return get_peft_model(base_model, peft_config)

    def forward(self, batch):
        outputs = self.esm_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
        
        # Get predictions from each head
        head_predictions = torch.cat([
            head(outputs.pooler_output) for head in self.ensemble_heads
        ], dim=1)  # Shape: (batch_size, num_heads)
        
        # Calculate mean and std across heads
        mean_pred = head_predictions.mean(dim=1, keepdim=True)
        std_pred = head_predictions.std(dim=1, keepdim=True)
        
        batch["head_predictions"] = head_predictions
        batch["predictions"] = mean_pred
        batch["uncertainties"] = std_pred
        batch['ucb'] = mean_pred + self.explore_weight * std_pred
        return batch

    def _compute_loss(self, head_predictions, targets):
        # Ensure targets have the same shape as predictions
        targets = targets.view(-1, 1)  # reshape to [batch_size, 1]
        
        # Compute BT loss for each head independently
        losses = []
        for i in range(head_predictions.shape[1]):  # For each head
            head_pred = head_predictions[:, i].view(-1, 1)  # reshape to [batch_size, 1]
            scores_i, scores_j, y_ij = create_pairwise_comparisons(
                {"predictions": head_pred, self.label: targets},
                head_pred,
                self.label
            )
            if scores_i is not None:
                head_loss = self.bt_loss(scores_i, scores_j, y_ij)
                losses.append(head_loss)
        
        # Return mean loss across heads if we have any valid comparisons
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.device, requires_grad=True)  # Add requires_grad=True

    def training_step(self, batch, batch_idx):
        batch = self(batch)
        head_predictions = batch["head_predictions"]
        mean_predictions = batch["predictions"]
        
        # Compute BT loss across all heads
        loss = self._compute_loss(head_predictions, batch[self.label])
        self.log("train_loss", loss)
        
        # Update metrics using mean prediction
        self.train_metrics(mean_predictions.view(-1), batch[self.label])
        self.log_dict(self.train_metrics)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self(batch)
        mean_predictions = batch["predictions"]
        uncertainties = batch["uncertainties"]
        
        # Log mean prediction metrics - ensure predictions maintain dimension
        self.val_mae(mean_predictions.view(-1), batch[self.label])
        self.val_spearman(mean_predictions.view(-1), batch[self.label])
        
        self.log("val_mae", self.val_mae, sync_dist=True)
        self.log("val_spearman", self.val_spearman, sync_dist=True)
        
        # Log average uncertainty
        self.log("val_uncertainty", uncertainties.mean(), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        return {
            'predictions': outputs['predictions'].view(-1),  # Changed from squeeze()
            'uncertainties': outputs['uncertainties'].view(-1),  # Changed from squeeze()
            'head_predictions': outputs['head_predictions'],  # Remove squeeze()
            'sequence': batch[self.hparams.xvar],
            'target': batch[self.label],
        }

    def save_model(self, save_path: str):
        """Save both PEFT adapter, ensemble heads, and pooler state"""
        adapter_state_dict = get_peft_model_state_dict(self.esm_model)
        ensemble_state_dict = self.ensemble_heads.state_dict()
        pooler_state_dict = self.esm_model.pooler.state_dict()  # Save pooler state

        save_dict = {
            'adapter_state_dict': adapter_state_dict,
            'ensemble_state_dict': ensemble_state_dict,
            'pooler_state_dict': pooler_state_dict,  # Include pooler state
            'hparams': dict(self.hparams),
            'model_name': self.hparams.model_name,
        }

        torch.save(save_dict, save_path)
        print(f"Model saved to: {save_path}")
        print(f"Adapter state dict keys: {list(adapter_state_dict.keys())}")
        print(f"Ensemble state dict keys: {list(ensemble_state_dict.keys())}")
        print(f"Pooler state dict keys: {list(pooler_state_dict.keys())}")

    @classmethod
    def load_model(cls, load_path: str):
        """Load a saved model including PEFT adapter, ensemble heads, and pooler state"""
        saved_dict = torch.load(load_path)

        # Initialize model with saved hyperparameters
        model = cls(**saved_dict['hparams'])

        # Load PEFT adapter
        set_peft_model_state_dict(model.esm_model, saved_dict['adapter_state_dict'])

        # Load ensemble heads
        model.ensemble_heads.load_state_dict(saved_dict['ensemble_state_dict'])

        # Load pooler state
        model.esm_model.pooler.load_state_dict(saved_dict['pooler_state_dict'])

        # Set to eval mode
        model.eval()

        print(f"Model loaded from: {load_path}")
        print(f"Loaded adapter state dict keys: {list(saved_dict['adapter_state_dict'].keys())}")
        print(f"Loaded ensemble state dict keys: {list(saved_dict['ensemble_state_dict'].keys())}")
        print(f"Loaded pooler state dict keys: {list(saved_dict['pooler_state_dict'].keys())}")

        return model

    @classmethod
    def load_from_wandb(cls, run_path: str, model_artifact_name: str, run = None):
        """Load model from wandb artifact"""
        if run is None:
            api = wandb.Api()
            artifact = api.artifact(f'{run_path}/{model_artifact_name}:latest')
            artifact_dir = artifact.download()
        else:
            artifact = run.use_artifact(f'{model_artifact_name}:latest')
            artifact_dir = artifact.download()

        model_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        model = cls.load_model(model_path)
        model.current_run_path = f'{artifact.entity}/{artifact.project}/{artifact.source_name}'
        return model
