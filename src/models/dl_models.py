"""
Classical deep learning classifier heads (CNN / RNN / LSTM / BiLSTM / GRU) that
operate on pre-computed, fixed-size embedding vectors (from TF-IDF or a frozen
transformer). Each model takes a single embedding vector per example and
outputs class logits.

Note: since the input here is a single pooled vector rather than a token
sequence, the "sequence" models (RNN/LSTM/BiLSTM/GRU) treat the embedding
dimension as a length-1 sequence for architectural consistency with the
paper's experimental setup, and the CNN uses 1D convolutions over the
embedding vector reshaped into a pseudo-sequence.
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_filters: int = 100, kernel_sizes=(2, 3, 4), dropout: float = 0.3):
        super().__init__()
        # Treat the embedding vector as a 1-channel sequence of length input_dim
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        concat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(concat))


class _RecurrentClassifier(nn.Module):
    """Shared scaffold for RNN / LSTM / GRU / BiLSTM heads."""

    def __init__(self, cell: str, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 1, dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        cell = cell.lower()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=1,  # each "timestep" is one scalar of the embedding vector
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, input_dim, 1) treated as a sequence
        x = x.unsqueeze(-1)
        output, hidden = self.rnn(x)
        last_step = output[:, -1, :]  # (batch, hidden_dim * num_directions)
        return self.fc(self.dropout(last_step))


class RNNClassifier(_RecurrentClassifier):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3):
        super().__init__("rnn", input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional=False)


class LSTMClassifier(_RecurrentClassifier):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3):
        super().__init__("lstm", input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional=False)


class BiLSTMClassifier(_RecurrentClassifier):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3):
        super().__init__("lstm", input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional=True)


class GRUClassifier(_RecurrentClassifier):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3):
        super().__init__("gru", input_dim, hidden_dim, num_classes, num_layers, dropout, bidirectional=False)


MODEL_REGISTRY = {
    "cnn": CNNClassifier,
    "rnn": RNNClassifier,
    "lstm": LSTMClassifier,
    "bilstm": BiLSTMClassifier,
    "gru": GRUClassifier,
}


def build_model(name: str, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.3):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")

    if name == "cnn":
        return CNNClassifier(input_dim=input_dim, num_classes=num_classes, dropout=dropout)

    return MODEL_REGISTRY[name](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
    )
