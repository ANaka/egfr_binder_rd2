from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from typing import Optional



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
        
    def setup(self, stage=None):
        # Convert training DataFrame to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(self.df)
        
        if self.val_df is not None:
            # Use provided validation split
            val_dataset = Dataset.from_pandas(self.val_df)
            test_dataset = val_dataset  # Use validation data as test set for simplicity
        else:
            # Perform random splits if validation data not provided
            splits = train_dataset.train_test_split(test_size=self.test_size, seed=self.seed)
            train_val = splits['train']
            test_dataset = splits['test']
            
            splits = train_val.train_test_split(test_size=self.val_size, seed=self.seed)
            train_dataset = splits['train']
            val_dataset = splits['test']

        # Transform yvar based on specified method
        if self.transform_type == 'standardize':
            # Calculate mean and std of yvar from the training dataset only
            self.y_mean = float(self.df[self.yvar].mean())
            self.y_std = float(self.df[self.yvar].std())
            
            # Standardize all datasets using training statistics
            def standardize(x):
                return {self.yvar: (x[self.yvar] - self.y_mean) / self.y_std}
            
            train_dataset = train_dataset.map(standardize)
            val_dataset = val_dataset.map(standardize)
            test_dataset = test_dataset.map(standardize)
        
        elif self.transform_type == 'rank':
            # Convert to ranks (0 to 1) for each dataset separately
            def rank_transform(dataset):
                ranks = pd.Series(dataset[self.yvar]).rank()
                ranks = ranks / len(ranks)  # normalize to [0,1]
                return dataset.map(
                    lambda x, idx: {self.yvar: float(ranks.iloc[idx])},
                    with_indices=True
                )
            
            train_dataset = rank_transform(train_dataset)
            val_dataset = rank_transform(val_dataset)
            test_dataset = rank_transform(test_dataset)

        if self.make_negative:
            def negate(x):
                if self.transform_type == 'standardize':
                    return {self.yvar: -1 * x[self.yvar]}
                else:  # rank
                    return {self.yvar: 1 - x[self.yvar]}
                    
            train_dataset = train_dataset.map(negate)
            val_dataset = val_dataset.map(negate)
            test_dataset = test_dataset.map(negate)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.xvar], 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length
            )

        # Apply tokenization to all datasets
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)
        self.test_dataset = test_dataset.map(tokenize_function, batched=True)

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
