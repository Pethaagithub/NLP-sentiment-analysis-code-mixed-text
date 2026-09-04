"""
Generic training loop for classical DL heads (CNN/RNN/LSTM/BiLSTM/GRU) trained
on top of fixed, pre-computed embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import EmbeddingDataset
from src.evaluate import compute_metrics


def train_dl_model(model: nn.Module, train_emb: np.ndarray, train_labels: np.ndarray,
                    val_emb: np.ndarray, val_labels: np.ndarray, device: str,
                    epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    train_loader = DataLoader(EmbeddingDataset(train_emb, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_emb, val_labels), batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        val_metrics, _, _ = evaluate_dl_model(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | train_loss={avg_train_loss:.4f} | "
              f"val_f1_macro={val_metrics['f1_macro']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def evaluate_dl_model(model: nn.Module, data_loader: DataLoader, device: str):
    model.eval()
    all_preds, all_labels = [], []

    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return metrics, np.array(all_labels), np.array(all_preds)
