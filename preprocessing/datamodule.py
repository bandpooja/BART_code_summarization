import multiprocessing
from preprocessing.dataset import CodeSearchNetBERTDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class CodeSearchNetBERTModule(pl.LightningDataModule):
    """
        pytorch-lightning data-module
    """
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=16, max_token_len=40):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

        # setting the datasets to None
        self.train_dataset = None
        self.encoding = None
        self.val_dataset = None
        self.test_dataset = None
        
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        if len(available_gpus) > 0:
            self.workers = len(available_gpus)
        else:
            self.workers = multiprocessing.cpu_count()

    def setup(self, stage=None):
        self.train_dataset = CodeSearchNetBERTDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.encoding = self.train_dataset.return_encoding()

        self.val_dataset = CodeSearchNetBERTDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len,
            encoding=self.encoding
        )

        self.test_dataset = CodeSearchNetBERTDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len,
            encoding=self.encoding
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers  # feed more than one batch at a time
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers  # feed more than one batch at a time
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers  # feed more than one batch at a time
        )
