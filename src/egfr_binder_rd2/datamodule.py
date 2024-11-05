from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from typing import Optional
import torch
import numpy as np



class SequenceDataModule(LightningDataModule):
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer_name: str, 
        yvar: str, 
        val_df: Optional[pd.DataFrame] = None,  # New parameter for pre-split validation data
        batch_size: int = 32, 
        max_length: int = 512, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        seed: int = 42, 
        xvar: str = 'binder_sequence', 
        transform_type: str = 'rank', 
        make_negative: bool = False
    ):
        super().__init__()
        self.df = df
        
        # Drop rows where yvar is not a float
        self.df = self.df[pd.to_numeric(self.df[yvar], errors='coerce').notna()].copy()
        
        self.val_df = val_df  # Store validation df if provided
        self.tokenizer_name = tokenizer_name
        self.yvar = yvar
        self.batch_size = batch_size
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.tokenizer = None
        self.xvar = xvar
        self.transform_type = transform_type
        self.y_mean = None
        self.y_std = None
        self.y_ranks = None
        self.make_negative = make_negative
        
        # Add new attributes to store split indices and DataFrames
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def setup(self, stage=None):
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # First split the data into train/val/test
        train_size = int(0.8 * len(self.df))
        val_size = int(0.1 * len(self.df))
        test_size = len(self.df) - train_size - val_size
        
        splits = torch.utils.data.random_split(
            self.df,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Store indices for each split
        self.train_indices = splits[0].indices
        self.val_indices = splits[1].indices
        self.test_indices = splits[2].indices
        
        # Store DataFrames for each split
        self.train_df = self.df.iloc[self.train_indices].copy()
        self.val_df = self.df.iloc[self.val_indices].copy()
        self.test_df = self.df.iloc[self.test_indices].copy()
        
        # Convert splits to datasets
        train_dataset = Dataset.from_pandas(self.train_df)
        val_dataset = Dataset.from_pandas(self.val_df)
        test_dataset = Dataset.from_pandas(self.test_df)

        # Combined transformation and tokenization function
        def process_dataset(dataset):
            def transform_and_tokenize(examples, indices):
                results = {}
                
                # Transform y values
                if self.transform_type == 'standardize':
                    # Convert list to numpy array for vectorized operations
                    y_values = np.array(examples[self.yvar])
                    standardized = (y_values - self.y_mean) / self.y_std
                    results[self.yvar] = standardized.tolist()  # Convert back to list
                elif self.transform_type == 'rank':
                    ranks = pd.Series(dataset[self.yvar]).rank()
                    normalized_ranks = ranks / len(ranks)
                    results[self.yvar] = [float(normalized_ranks.iloc[i]) for i in indices]
                
                # Apply negation if required
                if self.make_negative:
                    if self.transform_type == 'standardize':
                        results[self.yvar] = [-1 * y for y in results[self.yvar]]
                    else:  # rank
                        results[self.yvar] = [1 - y for y in results[self.yvar]]
                
                # Tokenization
                tokenized = self.tokenizer(
                    examples[self.xvar], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=self.max_length
                )
                results.update(tokenized)
                
                return results

            return dataset.map(
                transform_and_tokenize,
                batched=True,
                with_indices=True
            )

        # Calculate standardization parameters if needed
        if self.transform_type == 'standardize':
            self.y_mean = float(self.df[self.yvar].mean())
            self.y_std = float(self.df[self.yvar].std())

        # Process all datasets with a single map operation each
        self.train_dataset = process_dataset(train_dataset)
        self.val_dataset = process_dataset(val_dataset)
        self.test_dataset = process_dataset(test_dataset)

        # Set the format to PyTorch tensors
        columns = ["input_ids", "attention_mask", self.yvar, self.xvar]
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)
        self.test_dataset.set_format(type="torch", columns=columns)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

    @property
    def splits(self):
        """Return a dictionary containing the split DataFrames"""
        return {
            'train': self.train_df,
            'val': self.val_df,
            'test': self.test_df
        }
    
    @property
    def indices(self):
        """Return a dictionary containing the split indices"""
        return {
            'train': self.train_indices,
            'val': self.val_indices,
            'test': self.test_indices
        }

class SequenceFeaturesDataModule(SequenceDataModule):
    """DataModule for sequence data with additional numerical features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numerical_features = [
            'length',
            'perc_charged',
            'perc_hydrophobic',
            'p_soluble'
        ]
        
    def setup(self, stage: Optional[str] = None):
        # Call parent setup to handle sequence tokenization
        super().setup(stage)
        
        # Extract additional features for each split
        for split_name, split_df in self.splits.items():
            # Convert numerical features to tensor
            features = torch.tensor(
                split_df[self.numerical_features].values, 
                dtype=torch.float32
            )
            
            # Get corresponding dataset
            dataset = getattr(self, f"{split_name}_dataset")
            if dataset is not None:
                # Remove sequence column if it exists
                if 'sequence' in dataset.column_names:
                    dataset = dataset.remove_columns('sequence')
                
                # Add features and sequence columns
                dataset = dataset.add_column('numerical_features', features.tolist())
                dataset = dataset.add_column('sequence', split_df[self.xvar].tolist())
                
                # Update format to include numerical features and sequence
                columns = ["input_ids", "attention_mask", "numerical_features", self.yvar, "sequence"]
                dataset.set_format(type="torch", columns=columns)
                
                # Update the dataset reference
                setattr(self, f"{split_name}_dataset", dataset)