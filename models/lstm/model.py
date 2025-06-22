import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class StockDatasetLSTM(Dataset):
    """Dataset class for LSTM model that handles stock data with company embeddings."""

    def __init__(self, X: np.ndarray, y: np.ndarray, company_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.company_ids = torch.tensor(company_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], self.company_ids[idx]


class ModelLSTM(nn.Module):
    """LSTM model for stock price prediction with optional company embeddings."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        use_batch_norm: bool = True,
        num_companies: int = 0,
        embedding_dim: int = 8,
    ):
        super(ModelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_company_embeddings = num_companies > 0

        if self.use_company_embeddings:
            self.company_embedding = nn.Embedding(num_companies, embedding_dim)
            lstm_input_size = num_features + embedding_dim
        else:
            lstm_input_size = num_features

        self.lstm = nn.LSTM(
            lstm_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, company_ids=None):
        """Forward pass with optional company embeddings."""
        if self.use_company_embeddings and company_ids is not None:
            embeds = self.company_embedding(company_ids)  # [batch_size, embedding_dim]
            seq_len = x.size(1)
            embeds = embeds.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embedding_dim]

            x = torch.cat([x, embeds], dim=2)  # [batch_size, seq_len, features+embedding_dim]

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output

        if self.use_batch_norm:
            out = self.bn(out)

        out = self.dropout(out)
        out = self.fc(out)
        return out
