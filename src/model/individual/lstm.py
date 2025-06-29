import warnings

import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.config import EARLY_STOPPAGE, EPOCHS
from src.model.individual.model_utils import EarlyStopping, create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        # Batch normalization after LSTM
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Fully connected layers with batch norm
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Use the last time step output
        last_output = lstm_out[:, -1, :]

        # Apply batch normalization
        last_output = self.batch_norm(last_output)

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output


def train_lstm_model(
    X_train,
    y_train,
    X_test,
    y_test,
    input_size,
    epochs=EPOCHS,
    batch_size=32,
    learning_rate=0.001,
    early_stopping_patience=EARLY_STOPPAGE,
    weight_decay=0.01,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5,
    lr_scheduler_min_lr=1e-6,
    max_grad_norm=1.0,
):
    # Create datasets and dataloaders using common utility
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size, model_type='rnn')

    # Initialize model
    model = LSTM(input_size=input_size)
    criterion = nn.MSELoss()

    # Add L2 regularization via weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Add ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min_lr,
        verbose=True,
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, restore_best_weights=True)

    # Training loop
    train_losses = []
    test_losses = []

    print('\nTraining LSTM model...')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'Initial learning rate: {learning_rate}')
    print(f'L2 regularization (weight_decay): {weight_decay}')
    print(f'Gradient clipping max norm: {max_grad_norm}')
    print('Dropout rate: 0.3')
    print(f'LR scheduler patience: {lr_scheduler_patience} epochs')
    print(f'Early stopping patience: {early_stopping_patience} epochs')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch [{epoch + 1:3d}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}'
        )

        # Step the learning rate scheduler
        scheduler.step(test_loss)

        # Early stopping check
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            print(f'Best validation loss: {early_stopping.best_loss:.4f}')
            print(f'Final learning rate: {current_lr:.6f}')
            break

    # Restore best model weights
    early_stopping.restore_best_weights_to_model(model)
    final_lr = optimizer.param_groups[0]['lr']
    print(f'Restored best model weights (validation loss: {early_stopping.best_loss:.4f})')
    print(f'Training completed with final learning rate: {final_lr:.6f}')

    return model, train_losses, test_losses


def main(
    sequence_length=None,
    auto_sequence_length=True,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPAGE,
    weight_decay=0.01,
    lr_scheduler_patience=5,
    max_grad_norm=1.0,
):
    print('=' * 60)
    print('LSTM Stock Price Prediction - GPW Dataset')
    print('=' * 60)

    print('\nLoading dataset...')
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')

    print(f'Dataset shape: {df.shape}')
    print(f'Date range: {df["end_of_period"].min()} to {df["end_of_period"].max()}')
    print(f'Unique companies: {df["ticker"].nunique()}')

    # Prepare data using common utility
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_data(
        df,
        sequence_length=sequence_length,
        test_start_year=2021,
        auto_sequence_length=auto_sequence_length,
        model_type='rnn',
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print('ERROR: No sequences created! Check your data and sequence length.')
        return None, None

    # Train model
    input_size = len(feature_cols)
    print(f'\nModel input size: {input_size} features')

    model, train_losses, test_losses = train_lstm_model(
        X_train,
        y_train,
        X_test,
        y_test,
        input_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        weight_decay=weight_decay,
        lr_scheduler_patience=lr_scheduler_patience,
        max_grad_norm=max_grad_norm,
    )

    # Evaluate model using common utility
    results = evaluate_model(model, X_test, y_test, scaler_y, model_type='rnn', model_name='LSTM Model')

    return model, results


if __name__ == '__main__':
    model, results = main()
