import warnings

import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.config import EARLY_STOPPAGE, EPOCHS
from src.model.individual.model_utils import EarlyStopping, create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class SimpleCNN(nn.Module):
    def __init__(self, input_size, sequence_length, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length

        # Multiple 1D convolutional layers with different kernel sizes
        # to capture patterns at different time scales

        # First conv layer - captures short-term patterns
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)

        # Second conv layer - captures medium-term patterns
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Third conv layer - captures longer-term patterns
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2, padding=0)

        # Pooling and normalization layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        # Calculate the size after convolutions and pooling
        # This is a bit complex, so we'll use adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # Reduces to size 1

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, features, time_steps)

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Apply pooling only if sequence is long enough
        if x.size(2) > 1:
            x = self.pool(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Apply pooling only if sequence is long enough
        if x.size(2) > 1:
            x = self.pool(x)

        # Third conv block
        if x.size(2) >= 2:  # Only apply if we have enough time steps
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)  # Shape: (batch_size, 64, 1)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64)

        # Fully connected layers
        output = self.fc(x)

        return output


def train_cnn_model(
    X_train,
    y_train,
    X_test,
    y_test,
    input_size,
    sequence_length,
    epochs=EPOCHS,
    batch_size=32,
    learning_rate=0.001,
    early_stopping_patience=EARLY_STOPPAGE,
):
    # Create datasets and dataloaders using common utility
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size, model_type='cnn')

    # Initialize model
    model = SimpleCNN(input_size=input_size, sequence_length=sequence_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, restore_best_weights=True)

    # Training loop
    train_losses = []
    test_losses = []

    print('\nTraining CNN model...')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
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

        print(f'Epoch [{epoch + 1:3d}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

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
    print('=' * 60)
    print('CNN Stock Price Prediction - GPW Dataset')
    print('=' * 60)

    print('\nCNN Architecture Notes:')
    print('• Using 1D convolutions for temporal pattern detection')
    print('• Multiple kernel sizes to capture different time scales')
    print('• Requires minimum 3 time steps for meaningful convolutions')
    print('• May work better with longer sequences than RNNs')

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
        model_type='cnn',
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print('ERROR: No sequences created! Check your data and sequence length.')
        return None, None

    # Train model
    input_size = len(feature_cols)
    sequence_length_used = X_train.shape[1]
    print(f'\nModel input size: {input_size} features')
    print(f'Sequence length: {sequence_length_used} time steps')

    model, train_losses, test_losses = train_cnn_model(
        X_train,
        y_train,
        X_test,
        y_test,
        input_size,
        sequence_length_used,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
    )

    # Evaluate model using common utility
    results = evaluate_model(model, X_test, y_test, scaler_y, model_type='cnn', model_name='CNN Model')

    return model, results


if __name__ == '__main__':
    model, results = main()
