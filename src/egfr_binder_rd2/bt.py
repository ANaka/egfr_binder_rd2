import torch
import lightning as pl
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, SpearmanCorrCoef
from transformers import EsmForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict, get_peft_model_state_dict
import wandb
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Utility functions
def create_pairwise_comparisons(batch, outputs, label):
    device = outputs.device
    n_samples = outputs.shape[0]

    if n_samples < 2:
        return None, None, None

    pairs = torch.combinations(torch.arange(n_samples), r=2)
    labels = batch[label].to(device)

    scores_i = outputs[pairs[:, 0]]
    scores_j = outputs[pairs[:, 1]]

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
            self.log("train_bt_loss", bt_loss)
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

    def predict_step(self, batch, batch_idx):
        return self(batch)

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
