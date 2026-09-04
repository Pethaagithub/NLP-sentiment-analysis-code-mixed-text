"""
Dataset utilities: loading processed CSVs, label encoding, and PyTorch Dataset
wrappers for both the "frozen embedding + DL head" path and the
"end-to-end transformer fine-tuning" path.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def load_split(processed_dir: str, lang: str, split: str) -> pd.DataFrame:
    """Load a processed CSV split, e.g. data/processed/tamil_train.csv"""
    path = f"{processed_dir}/{lang}_{split}.csv"
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), f"{path} must have 'text' and 'label' columns"
    return df


def fit_label_encoder(train_df: pd.DataFrame) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(train_df["label"])
    return le


class EmbeddingDataset(Dataset):
    """
    Wraps pre-computed embeddings (numpy array, one row per example) with integer labels.
    Used for the CNN / RNN / LSTM / BiLSTM / GRU heads trained on top of frozen embeddings.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        assert len(embeddings) == len(labels)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


class TransformerTextDataset(Dataset):
    """
    Tokenizes raw text on the fly for end-to-end transformer fine-tuning
    (mBERT / MuRIL / IndicBERT / XLM-RoBERTa).
    """

    def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item
