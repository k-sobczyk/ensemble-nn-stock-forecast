import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        # Fully connected layers
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Use the last time step output
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output


def analyze_sequence_length_coverage(data, ticker_col, date_col, test_start_year=2021):
    # Analyze coverage for both training and test periods
    train_data = data[pd.to_datetime(data[date_col]).dt.year < test_start_year]
    test_data = data[pd.to_datetime(data[date_col]).dt.year >= test_start_year]
    
    train_lengths = []
    test_lengths = []
    
    for ticker in data[ticker_col].unique():
        # Training period lengths
        train_ticker_data = train_data[train_data[ticker_col] == ticker]
        if len(train_ticker_data) > 0:
            train_lengths.append(len(train_ticker_data))
        
        # Test period lengths  
        test_ticker_data = test_data[test_data[ticker_col] == ticker]
        if len(test_ticker_data) > 0:
            test_lengths.append(len(test_ticker_data))

    print(f"Training period: {len(train_lengths)} companies")
    print(f"Test period: {len(test_lengths)} companies")
    print(f"Average training periods per company: {np.mean(train_lengths):.1f}")
    print(f"Average test periods per company: {np.mean(test_lengths):.1f}")
    
    sequence_coverage = {}
    for seq_len in range(1, 13):
        train_companies = sum(1 for length in train_lengths if length > seq_len)
        test_companies = sum(1 for length in test_lengths if length > seq_len)
        
        train_coverage = (train_companies / len(train_lengths)) * 100 if train_lengths else 0
        test_coverage = (test_companies / len(test_lengths)) * 100 if test_lengths else 0
        
        sequence_coverage[seq_len] = {
            'train_companies': train_companies,
            'test_companies': test_companies,
            'train_coverage': train_coverage,
            'test_coverage': test_coverage
        }

        if seq_len <= 10:
            print(
                f'Sequence length {seq_len:2d}: Train {train_companies:3d}/{len(train_lengths)} ({train_coverage:5.1f}%), '
                f'Test {test_companies:3d}/{len(test_lengths)} ({test_coverage:5.1f}%)'
            )

    # Find optimal sequence length (where at least 80% of companies can contribute in BOTH periods)
    optimal_seq_len = 3  # conservative default
    for seq_len in range(1, 13):
        train_cov = sequence_coverage[seq_len]['train_coverage']
        test_cov = sequence_coverage[seq_len]['test_coverage']
        if train_cov >= 80 and test_cov >= 80:
            optimal_seq_len = seq_len
        else:
            break

    print(
        f'\nRecommended sequence length: {optimal_seq_len}'
        f' (Train: {sequence_coverage[optimal_seq_len]["train_coverage"]:.1f}%, '
        f'Test: {sequence_coverage[optimal_seq_len]["test_coverage"]:.1f}%)'
    )
    return optimal_seq_len


def create_sequences(data, target, ticker_col, date_col, sequence_length):
    sequences = []
    targets = []
    companies_included = 0
    companies_skipped = 0

    # Group by ticker to create sequences for each company
    for ticker in data[ticker_col].unique():
        ticker_data = data[data[ticker_col] == ticker].sort_values(date_col).reset_index(drop=True)

        # Skip companies with insufficient data
        if len(ticker_data) <= sequence_length:
            companies_skipped += 1
            continue

        companies_included += 1

        # Create sequences for this ticker
        for i in range(len(ticker_data) - sequence_length):
            # Get feature columns (all except target_log)
            feature_cols = [col for col in ticker_data.columns if col not in ['target_log', 'ticker', 'end_of_period']]
            seq = ticker_data.iloc[i : i + sequence_length][feature_cols].values
            tgt = ticker_data.iloc[i + sequence_length]['target_log']

            # Only add if no NaN values
            if not np.isnan(seq).any() and not np.isnan(tgt):
                sequences.append(seq)
                targets.append(tgt)

    print(f'Companies included: {companies_included}, skipped: {companies_skipped}')
    return np.array(sequences), np.array(targets)


def prepare_data(df, sequence_length=None, test_start_year=2021, auto_sequence_length=True):
    df['end_of_period'] = pd.to_datetime(df['end_of_period'])
    df = df.sort_values(['ticker', 'end_of_period']).reset_index(drop=True)

    # Define feature columns (exclude target, ticker, and date)
    feature_cols = [col for col in df.columns if col not in ['target_log', 'ticker', 'end_of_period']]

    # Analyze sequence length coverage if auto mode is enabled
    if auto_sequence_length or sequence_length is None:
        print('Analyzing optimal sequence length...')
        sequence_length = analyze_sequence_length_coverage(df, 'ticker', 'end_of_period', test_start_year)

    print(f'Using sequence length: {sequence_length}')

    # Split into train and test based on year
    train_df = df[df['end_of_period'].dt.year < test_start_year].copy()
    test_df = df[df['end_of_period'].dt.year >= test_start_year].copy()

    print(f'\nTraining data: {train_df.shape[0]} samples')
    print(f'Test data: {test_df.shape[0]} samples')
    print(f'Training date range: {train_df["end_of_period"].min()} to {train_df["end_of_period"].max()}')
    print(f'Test date range: {test_df["end_of_period"].min()} to {test_df["end_of_period"].max()}')

    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit scalers on training data
    train_df[feature_cols] = scaler_X.fit_transform(train_df[feature_cols])
    train_target_scaled = scaler_y.fit_transform(train_df[['target_log']])
    train_df['target_log'] = train_target_scaled.flatten()

    # Transform test data
    test_df[feature_cols] = scaler_X.transform(test_df[feature_cols])
    test_target_scaled = scaler_y.transform(test_df[['target_log']])
    test_df['target_log'] = test_target_scaled.flatten()

    # Create sequences
    print(f'\nCreating sequences with length {sequence_length}...')
    print('Training set:')
    X_train, y_train = create_sequences(train_df, train_df['target_log'], 'ticker', 'end_of_period', sequence_length)
    print('Test set:')
    X_test, y_test = create_sequences(test_df, test_df['target_log'], 'ticker', 'end_of_period', sequence_length)

    print('\nFinal sequences:')
    print(f'Training sequences: {X_train.shape}')
    print(f'Test sequences: {X_test.shape}')

    return X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols


def train_lstm_model(X_train, y_train, X_test, y_test, input_size, epochs=50, batch_size=32, learning_rate=0.001):
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleLSTM(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []

    print('\nTraining LSTM model...')

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


def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.FloatTensor(X_test)).numpy()

    # Inverse transform predictions and targets
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler_y.inverse_transform(test_predictions).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    rmse = np.sqrt(mse)

    print('\nModel Performance:')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': predictions_original, 'actual': y_test_original}


def main(sequence_length=None, auto_sequence_length=True, epochs=50):
    print('=' * 60)
    print('LSTM Stock Price Prediction - GPW Dataset')
    print('=' * 60)

    print('\nLoading dataset...')
    df = pd.read_csv('src/model/individual/dataset_1_full_features.csv')

    print(f'Dataset shape: {df.shape}')
    print(f'Date range: {df["end_of_period"].min()} to {df["end_of_period"].max()}')
    print(f'Unique companies: {df["ticker"].nunique()}')

    # Prepare data with adaptive sequence length
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_data(
        df, sequence_length=sequence_length, test_start_year=2021, auto_sequence_length=auto_sequence_length
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print('ERROR: No sequences created! Check your data and sequence length.')
        return None, None

    # Train model
    input_size = len(feature_cols)
    print(f'\nModel input size: {input_size} features')

    model, train_losses, test_losses = train_lstm_model(X_train, y_train, X_test, y_test, input_size, epochs=epochs)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test, scaler_y)

    return model, results


if __name__ == '__main__':
    model, results = main()
