import warnings

import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.model_utils import create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class SimpleBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        self.bi_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,  # Key difference: bidirectional processing
        )

        # Fully connected layers
        # Note: bidirectional LSTM outputs hidden_size * 2 (forward + backward)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        # Initialize hidden state for bidirectional LSTM
        batch_size = x.size(0)
        # For bidirectional LSTM, we need num_layers * 2 (forward + backward)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)

        # Bidirectional LSTM forward pass
        lstm_out, _ = self.bi_lstm(x, (h0, c0))

        # Use the last time step output (contains both forward and backward info)
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output


def train_bi_lstm_model(X_train, y_train, X_test, y_test, input_size, epochs=50, batch_size=32, learning_rate=0.001):
    # Create datasets and dataloaders using common utility
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size, model_type='rnn')

    # Initialize model
    model = SimpleBiLSTM(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []

    print('\nTraining Bidirectional LSTM model...')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

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

    return model, train_losses, test_losses


def main(sequence_length=None, auto_sequence_length=True, epochs=50):
    print('=' * 60)
    print('Bidirectional LSTM Stock Price Prediction - GPW Dataset')
    print('=' * 60)

    print('\nBi-LSTM Architecture Notes:')
    print('• Processes sequences in both forward and backward directions')
    print('• Can capture patterns that depend on both past and future context')
    print('• Output dimension is doubled (hidden_size * 2)')
    print('• Generally more powerful than unidirectional LSTM')

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

    model, train_losses, test_losses = train_bi_lstm_model(X_train, y_train, X_test, y_test, input_size, epochs=epochs)

    # Evaluate model using common utility
    results = evaluate_model(model, X_test, y_test, scaler_y, model_type='rnn', model_name='Bidirectional LSTM Model')

    return model, results


if __name__ == '__main__':
    model, results = main()
