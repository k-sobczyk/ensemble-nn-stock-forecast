import os
import warnings

import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.config import (
    EARLY_STOPPAGE,
    EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_HIDDEN_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_LR_SCHEDULER_FACTOR,
    LSTM_LR_SCHEDULER_PATIENCE,
    LSTM_MAX_GRAD_NORM,
    LSTM_NUM_LAYERS,
    LSTM_WEIGHT_DECAY,
)
from src.model.individual.model_utils import EarlyStopping, create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
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

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Initialize hidden state on the same device as input
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

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
    batch_size=LSTM_BATCH_SIZE,
    learning_rate=LSTM_LEARNING_RATE,
    early_stopping_patience=EARLY_STOPPAGE,
    weight_decay=LSTM_WEIGHT_DECAY,
    lr_scheduler_patience=LSTM_LR_SCHEDULER_PATIENCE,
    lr_scheduler_factor=LSTM_LR_SCHEDULER_FACTOR,
    lr_scheduler_min_lr=1e-6,
    max_grad_norm=LSTM_MAX_GRAD_NORM,
    hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT,
    device=None,
):
    # Check for CUDA availability
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')

    # Create datasets and dataloaders using common utility
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size, model_type='rnn')

    # Initialize model and move to device
    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()

    # Add L2 regularization via weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Add ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, min_lr=lr_scheduler_min_lr
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, restore_best_weights=True)

    # Training loop
    train_losses = []
    test_losses = []

    print('\nTraining LSTM model...')
    print(f'Architecture: Hidden={hidden_size}, Layers={num_layers}, Dropout={dropout}, LR={learning_rate}')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

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
                # Move data to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

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
            print(f'Early stopping at epoch {epoch + 1}, best loss: {early_stopping.best_loss:.4f}')
            break

    # Restore best model weights
    early_stopping.restore_best_weights_to_model(model)
    print(f'Training completed. Best validation loss: {early_stopping.best_loss:.4f}')

    return model, train_losses, test_losses


def main(
    sequence_length=None,
    auto_sequence_length=True,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPAGE,
):
    print('LSTM Stock Price Prediction')
    print('Loading dataset...')
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')
    print(f'Dataset: {df.shape[0]} samples, {df.shape[1]} features')

    # Prepare data using common utility
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

    # Get input size
    input_size = len(feature_cols)
    sequence_length_used = X_train.shape[1]
    print(f'Input: {input_size} features, {sequence_length_used} time steps')

    # Train model with optimized parameters from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_losses, test_losses = train_lstm_model(
        X_train,
        y_train,
        X_test,
        y_test,
        input_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )

    # Save loss plot
    os.makedirs('src/model/individual/output/lstm_results', exist_ok=True)
    from src.model.individual.model_utils import plot_and_save_loss, save_individual_model_results

    plot_and_save_loss(train_losses, test_losses, 'src/model/individual/output/lstm_results/lstm_loss.png', 'LSTM')

    # Final evaluation
    print('\nFinal evaluation:')
    results = evaluate_model(model, X_test, y_test, scaler_y, model_type='rnn', model_name='LSTM')

    # Save comprehensive results including MAPE, MASE, and training data
    save_individual_model_results('LSTM', results, train_losses, test_losses, 'src/model/individual/output')

    return model, results


if __name__ == '__main__':
    model, results = main()
