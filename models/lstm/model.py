import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class StockDatasetLSTM(Dataset):
    """Dataset class for LSTM model that handles stock data with company embeddings."""
    def __init__(self, X: np.ndarray, y: np.ndarray, company_ids: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.company_ids = torch.tensor(company_ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx], self.company_ids[idx]


class ModelLSTM(nn.Module):
    """LSTM model for stock price prediction with optional company embeddings."""
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout_rate: float,
                 use_batch_norm: bool = True, num_companies: int = 0, embedding_dim: int = 8):
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

        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
