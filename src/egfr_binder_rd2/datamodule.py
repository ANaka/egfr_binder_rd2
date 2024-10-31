from datasets import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer



class SequenceDataModule(LightningDataModule):
    def __init__(self, df, tokenizer_name, yvar, batch_size=32, max_length=512, test_size=0.2, val_size=0.1, 
                 seed=42, xvar='Sequence', transform_type='rank', make_negative:bool=False):
        super().__init__()
        self.df = df
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
        # Convert DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(self.df)

        # Transform yvar based on specified method
        if self.transform_type == 'standardize':
            # Calculate mean and std of yvar from the entire dataset
            self.y_mean = float(self.df[self.yvar].mean())
            self.y_std = float(self.df[self.yvar].std())
            # Standardize yvar
            dataset = dataset.map(lambda x: {self.yvar: (x[self.yvar] - self.y_mean) / self.y_std})
        
        elif self.transform_type == 'rank':
            # Convert to ranks (0 to 1)
            ranks = self.df[self.yvar].rank()
            self.y_ranks = ranks / len(ranks)  # normalize to [0,1]
            dataset = dataset.map(lambda x, idx: {self.yvar: float(self.y_ranks.iloc[idx])}, 
                                with_indices=True)

        if self.make_negative:
            dataset = dataset.map(lambda x: {self.yvar: -1 * x[self.yvar]})

        # Split the dataset
        splits = dataset.train_test_split(test_size=self.test_size, seed=self.seed)
        train_val = splits['train']
        test = splits['test']

        # Further split train into train and validation
        splits = train_val.train_test_split(test_size=self.val_size, seed=self.seed)
        train = splits['train']
        val = splits['test']

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(examples[self.xvar], padding="max_length", truncation=True, max_length=self.max_length)

        # Apply tokenization
        self.train_dataset = train.map(tokenize_function, batched=True)
        self.val_dataset = val.map(tokenize_function, batched=True)
        self.test_dataset = test.map(tokenize_function, batched=True)

        # Set the format to PyTorch tensors
        columns = ["input_ids", "attention_mask", self.yvar]
        self.train_dataset.set_format(type="torch", columns=columns)
        self.val_dataset.set_format(type="torch", columns=columns)
        self.test_dataset.set_format(type="torch", columns=columns)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
