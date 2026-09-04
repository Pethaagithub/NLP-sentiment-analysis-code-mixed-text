"""
Embedding extraction strategies:
  - TF-IDF: sparse, non-contextual baseline features.
  - Transformer embeddings (mBERT / MuRIL / IndicBERT / XLM-RoBERTa): mean-pooled
    contextual embeddings, extracted with the encoder frozen, for use as fixed
    features feeding into CNN/RNN/LSTM/BiLSTM/GRU heads.

For end-to-end fine-tuning (rather than frozen features), see
`src/models/transformer_models.py`.
"""

from typing import List

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

TRANSFORMER_CHECKPOINTS = {
    "mbert": "bert-base-multilingual-cased",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
    "xlm_roberta": "xlm-roberta-large",
}


def fit_tfidf(train_texts: List[str], max_features: int = 20000, ngram_range=(1, 2)) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(train_texts)
    return vectorizer


def tfidf_transform(vectorizer: TfidfVectorizer, texts: List[str]) -> np.ndarray:
    return vectorizer.transform(texts).toarray()


class TransformerEmbedder:
    """
    Extracts frozen, mean-pooled sentence embeddings from a pretrained multilingual
    transformer. Useful for feeding classical DL heads (CNN/RNN/LSTM/BiLSTM/GRU)
    with contextual features, as opposed to fine-tuning the transformer end-to-end.
    """

    def __init__(self, model_key: str, device: str = None, max_length: int = 128):
        if model_key not in TRANSFORMER_CHECKPOINTS:
            raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(TRANSFORMER_CHECKPOINTS)}")
        checkpoint = TRANSFORMER_CHECKPOINTS[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state  # (batch, seq_len, hidden)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)

            # Mean pooling over non-padded tokens
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

            all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings)
