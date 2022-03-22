import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer, BartTokenizer


class CodeSearchNetBERTDataset(Dataset):
    """
        pytorch-lightning data-set
    """
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 40,
                 x_col: str = "code", y_col: str = "language", encoding: dict = None):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.x_col = x_col
        self.y_col = y_col
        self.uq_ys = np.unique(self.data[self.y_col].values.tolist())
        if encoding is None:
            self.encoding = {}
            for idx, lang_ in enumerate(self.uq_ys):
                self.encoding[lang_] = idx
        else:
            self.encoding = encoding

    def return_encoding(self):
        return self.encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        code = data_row[self.x_col]
        label = data_row[self.y_col]

        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,  # make sure that each sequence is of max_token_len
            return_attention_mask=True,
            return_tensors="pt"  # to return tensors like pytorch
        )

        return dict(
            code=code,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=torch.tensor(self.encoding[label])  # required by the loss function
        )


class CodeSearchNetBARTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BartTokenizer, max_token_len: int = 500,
                 x_col: str = 'code', y_col: str = 'summary'):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.x_col = x_col
        self.y_col = y_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        code = data_row[self.x_col]
        summary = data_row[self.y_col]

        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding=True,
            truncation=True,  # make sure that each sequence is of max_token_len
            return_attention_mask=True,
            return_tensors="pt"  # to return tensors like pytorch
        )
        encoding_summary = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            max_length=100,
            return_token_type_ids=False,
            padding=True,
            truncation=True,  # make sure that each sequence is of max_token_len
            return_attention_mask=True,
            return_tensors="pt"  # to return tensors like pytorch
        )
        return dict(
            # code=code,
            input_ids=encoding['input_ids'].flatten(),
            # attention_mask=encoding['attention_mask'].flatten(),
            # summary=summary,
            labels=encoding_summary['input_ids'].flatten()
        )
