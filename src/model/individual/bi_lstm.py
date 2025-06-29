import warnings

import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.config import (
    BI_LSTM_BATCH_SIZE,
    BI_LSTM_DROPOUT,
    BI_LSTM_HIDDEN_SIZE,
    BI_LSTM_LEARNING_RATE,
    BI_LSTM_LR_SCHEDULER_FACTOR,
    BI_LSTM_LR_SCHEDULER_PATIENCE,
    BI_LSTM_MAX_GRAD_NORM,
    BI_LSTM_NUM_LAYERS,
    BI_LSTM_WEIGHT_DECAY,
    EARLY_STOPPAGE,
    EPOCHS,
)
from src.model.individual.model_utils import EarlyStopping, create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class Bi_LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size=BI_LSTM_HIDDEN_SIZE, num_layers=BI_LSTM_NUM_LAYERS, dropout=BI_LSTM_DROPOUT
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Bidirectional LSTM layer with enhanced regularization
        self.bi_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # No dropout for single layer
            batch_first=True,
            bidirectional=True,  # Key difference: bidirectional processing
        )

        # Batch normalization for LSTM output stability
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional

        # Enhanced fully connected layers with proper regularization
        # Note: bidirectional LSTM outputs hidden_size * 2 (forward + backward)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state for bidirectional LSTM
        # For bidirectional LSTM, we need num_layers * 2 (forward + backward)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)

        # Bidirectional LSTM forward pass
        lstm_out, _ = self.bi_lstm(x, (h0, c0))

        # Use the last time step output (contains both forward and backward info)
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)

        # Apply batch normalization
        if last_output.size(0) > 1:  # Only apply if batch size > 1
            last_output = self.batch_norm(last_output)

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output


def train_bi_lstm_model(
    X_train,
    y_train,
    X_test,
    y_test,
    input_size,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPAGE,
):
    # Create datasets and dataloaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, BI_LSTM_BATCH_SIZE, model_type='rnn'
    )

    # Initialize model
    model = Bi_LSTM(input_size=input_size)
    criterion = nn.MSELoss()

    # Enhanced optimizer with L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=BI_LSTM_LEARNING_RATE, weight_decay=BI_LSTM_WEIGHT_DECAY)

    # Dynamic learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=BI_LSTM_LR_SCHEDULER_FACTOR,
        patience=BI_LSTM_LR_SCHEDULER_PATIENCE,
        min_lr=1e-6,
        verbose=True,
    )

    # Enhanced early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, restore_best_weights=True)

    # Training tracking
    train_losses = []
    test_losses = []

    print('\nTraining Optimized Bidirectional LSTM model...')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(
        f'Architecture: {BI_LSTM_NUM_LAYERS} layers, {BI_LSTM_HIDDEN_SIZE} hidden units, {BI_LSTM_DROPOUT:.2f} dropout'
    )
    print(f'Regularization: L2={BI_LSTM_WEIGHT_DECAY}, grad_clip={BI_LSTM_MAX_GRAD_NORM}')
    print(f'LR Schedule: patience={BI_LSTM_LR_SCHEDULER_PATIENCE}, factor={BI_LSTM_LR_SCHEDULER_FACTOR}')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), BI_LSTM_MAX_GRAD_NORM)

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()

        # Calculate average losses
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        # Update learning rate scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f'Epoch [{epoch + 1:3d}/{epochs}], Train Loss: {train_loss:.4f}, '
            f'Test Loss: {test_loss:.4f}, LR: {current_lr:.2e}'
        )

        # Early stopping check
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            print(f'Best validation loss: {early_stopping.best_loss:.4f}')
            break

    # Restore best model weights
    early_stopping.restore_best_weights_to_model(model)
    print(f'Restored best model weights (validation loss: {early_stopping.best_loss:.4f})')

    return model, train_losses, test_losses


def main(sequence_length=None, auto_sequence_length=True, epochs=EPOCHS, early_stopping_patience=EARLY_STOPPAGE):
    print('=' * 80)
    print('OPTIMIZED BIDIRECTIONAL LSTM STOCK PRICE PREDICTION - GPW DATASET')
    print('=' * 80)

    print('\n🏆 OPTIMIZED BI-LSTM ARCHITECTURE (VAL LOSS: 0.3374):')
    print('• Bidirectional processing (forward + backward)')
    print('• L2 regularization + enhanced dropout')
    print('• Batch normalization + gradient clipping')
    print('• Dynamic learning rate scheduling')
    print('• Hyperparameters optimized with Optuna')

    # Load and prepare data
    print('\n📊 Loading dataset...')
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')

    print(f'Dataset shape: {df.shape}')
    print(f'Date range: {df["end_of_period"].min()} to {df["end_of_period"].max()}')
    print(f'Unique companies: {df["ticker"].nunique()}')

    # Prepare data
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_data(
        df,
        sequence_length=sequence_length,
        test_start_year=2021,
        auto_sequence_length=auto_sequence_length,
        model_type='rnn',
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print('❌ ERROR: No sequences created! Check your data and sequence length.')
        return None, None

    input_size = len(feature_cols)
    print(f'\n🎯 Model input size: {input_size} features')

    # Train with optimized parameters
    print('\n🚀 Training with optimized parameters...')
    model, train_losses, test_losses = train_bi_lstm_model(
        X_train, y_train, X_test, y_test, input_size, epochs=epochs, early_stopping_patience=early_stopping_patience
    )

    # Comprehensive evaluation
    print('\n📈 FINAL MODEL EVALUATION:')
    results = evaluate_model(
        model, X_test, y_test, scaler_y, model_type='rnn', model_name='Optimized Bidirectional LSTM'
    )

    return model, results


if __name__ == '__main__':
    model, results = main()
