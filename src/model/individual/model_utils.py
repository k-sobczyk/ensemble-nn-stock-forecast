import copy
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Import metrics
from src.model.metrics.metrics import calculate_mape, calculate_mape_log_scale, calculate_mase, calculate_symmetric_mape

warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model):
        """Save model when validation loss decrease."""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())

    def restore_best_weights_to_model(self, model):
        """Load the best model weights."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class StockDataset(Dataset):
    """Common dataset class for all stock prediction models."""

    def __init__(self, sequences, targets, model_type='rnn'):
        self.sequences = sequences
        self.targets = targets
        self.model_type = model_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.model_type == 'cnn':
            # For CNN, we need to transpose to (features, time_steps) format
            sequence = torch.FloatTensor(self.sequences[idx]).transpose(0, 1)  # Shape: (features, time_steps)
        else:
            # For RNN-based models (LSTM, GRU, Bi-LSTM)
            sequence = torch.FloatTensor(self.sequences[idx])

        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target


def analyze_sequence_length_coverage(data, ticker_col, date_col, test_start_year=2021, min_seq_len=1):
    train_data = data[pd.to_datetime(data[date_col]).dt.year < test_start_year]
    test_data = data[pd.to_datetime(data[date_col]).dt.year >= test_start_year]

    train_lengths = []
    test_lengths = []

    for ticker in data[ticker_col].unique():
        train_ticker_data = train_data[train_data[ticker_col] == ticker]
        if len(train_ticker_data) > 0:
            train_lengths.append(len(train_ticker_data))

        # Test period lengths
        test_ticker_data = test_data[test_data[ticker_col] == ticker]
        if len(test_ticker_data) > 0:
            test_lengths.append(len(test_ticker_data))

    print(f'Training period: {len(train_lengths)} companies')
    print(f'Test period: {len(test_lengths)} companies')
    print(f'Average training periods per company: {np.mean(train_lengths):.1f}')
    print(f'Average test periods per company: {np.mean(test_lengths):.1f}')

    sequence_coverage = {}
    for seq_len in range(min_seq_len, 13):
        train_companies = sum(1 for length in train_lengths if length > seq_len)
        test_companies = sum(1 for length in test_lengths if length > seq_len)

        train_coverage = (train_companies / len(train_lengths)) * 100 if train_lengths else 0
        test_coverage = (test_companies / len(test_lengths)) * 100 if test_lengths else 0

        sequence_coverage[seq_len] = {
            'train_companies': train_companies,
            'test_companies': test_companies,
            'train_coverage': train_coverage,
            'test_coverage': test_coverage,
        }

        if seq_len <= 10:
            print(
                f'Sequence length {seq_len:2d}: Train {train_companies:3d}/{len(train_lengths)} ({train_coverage:5.1f}%), '
                f'Test {test_companies:3d}/{len(test_lengths)} ({test_coverage:5.1f}%)'
            )

    # Find optimal sequence length (where at least 80% of companies can contribute in BOTH periods)
    optimal_seq_len = min_seq_len  # conservative default
    for seq_len in range(min_seq_len, 13):
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

    if min_seq_len > 1:
        print(f'Note: Model requires minimum {min_seq_len} time steps')

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


def prepare_data(df, sequence_length=None, test_start_year=2021, auto_sequence_length=True, model_type='rnn'):
    df['end_of_period'] = pd.to_datetime(df['end_of_period'])
    df = df.sort_values(['ticker', 'end_of_period']).reset_index(drop=True)

    # Define feature columns (exclude target, ticker, and date)
    feature_cols = [col for col in df.columns if col not in ['target_log', 'ticker', 'end_of_period']]

    # Analyze sequence length coverage if auto mode is enabled
    if auto_sequence_length or sequence_length is None:
        # Set minimum sequence length based on model type
        min_seq_len = 3 if model_type == 'cnn' else 1

        print(f'Analyzing optimal sequence length for {model_type.upper()}...')
        sequence_length = analyze_sequence_length_coverage(df, 'ticker', 'end_of_period', test_start_year, min_seq_len)

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


def evaluate_model(model, X_test, y_test, scaler_y, model_type='rnn', model_name='Model'):
    model.eval()

    device = next(model.parameters()).device

    if model_type == 'cnn':
        # For CNN, use DataLoader to handle proper data transformation
        test_dataset = StockDataset(X_test, y_test, model_type='cnn')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        predictions = []
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            with torch.no_grad():
                batch_pred = model(batch_X).detach().cpu().numpy()
                predictions.extend(batch_pred)

        test_predictions = np.array(predictions)
    else:
        # For RNN-based models (LSTM, GRU, Bi-LSTM)
        with torch.no_grad():
            test_predictions = model(torch.FloatTensor(X_test).to(device)).detach().cpu().numpy()

    # Inverse transform predictions and targets
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler_y.inverse_transform(test_predictions).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    rmse = np.sqrt(mse)

    print(f'\n{model_name} Performance:')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': predictions_original, 'actual': y_test_original}


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, model_type='rnn'):
    train_dataset = StockDataset(X_train, y_train, model_type=model_type)
    test_dataset = StockDataset(X_test, y_test, model_type=model_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_and_save_loss(train_losses, test_losses, out_path, model_name='Model'):
    """Create and save training loss plot for any model."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_additional_metrics(actual, predicted):
    """Calculate MAPE and MASE metrics using standardized functions."""
    # MAPE - Mean Absolute Percentage Error (traditional)
    try:
        mape = calculate_mape(actual, predicted)
    except Exception as e:
        print(f'Warning: Could not calculate MAPE: {e}')
        mape = np.nan

    # Symmetric MAPE - More robust for stock prices
    try:
        smape = calculate_symmetric_mape(actual, predicted)
    except Exception as e:
        print(f'Warning: Could not calculate Symmetric MAPE: {e}')
        smape = np.nan

    # MAPE on log scale - More appropriate for log-transformed data
    try:
        mape_log = calculate_mape_log_scale(actual, predicted)
    except Exception as e:
        print(f'Warning: Could not calculate Log-scale MAPE: {e}')
        mape_log = np.nan

    # MASE - Mean Absolute Scaled Error
    # For MASE, we use actual values as training data for naive forecast
    try:
        mase = calculate_mase(actual, predicted, actual)
    except Exception as e:
        print(f'Warning: Could not calculate MASE: {e}')
        mase = np.nan

    return mape, mase, smape, mape_log


def save_individual_model_results(
    model_name, results, train_losses, test_losses, results_dir, save_detailed_plots=False
):
    """Save individual model results including metrics and loss data."""
    model_dir = os.path.join(results_dir, f'{model_name.lower()}_results')
    os.makedirs(model_dir, exist_ok=True)

    # Calculate additional metrics
    actual = results['actual']
    predicted = results['predictions']
    mape, mase, smape, mape_log = calculate_additional_metrics(actual, predicted)

    # Create comprehensive results dictionary
    comprehensive_results = {
        'model_name': model_name,
        'metrics': {
            'rmse': float(results['rmse']),
            'mae': float(results['mae']),
            'r2': float(results['r2']),
            'mape': float(mape),
            'smape': float(smape),
            'mape_log': float(mape_log),
            'mase': float(mase),
        },
        'training_data': {
            'train_losses': [float(loss) for loss in train_losses],
            'validation_losses': [float(loss) for loss in test_losses],
            'total_epochs': len(train_losses),
            'best_train_loss': float(min(train_losses)) if train_losses else 0,
            'best_validation_loss': float(min(test_losses)) if test_losses else 0,
        },
        'predictions': {'actual_values': [float(x) for x in actual], 'predicted_values': [float(x) for x in predicted]},
    }

    # Always save basic results
    # Save comprehensive results to JSON
    with open(os.path.join(model_dir, f'{model_name.lower()}_detailed_results.json'), 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([comprehensive_results['metrics']])
    metrics_df.to_csv(os.path.join(model_dir, f'{model_name.lower()}_metrics.csv'), index=False)

    # Only save detailed plots and CSVs for the best model
    if save_detailed_plots:
        print(f'🏆 Saving detailed visualizations for BEST MODEL: {model_name}')

        # Save loss data to CSV
        if train_losses and test_losses:
            loss_df = pd.DataFrame(
                {'epoch': range(1, len(train_losses) + 1), 'train_loss': train_losses, 'validation_loss': test_losses}
            )
            loss_df.to_csv(os.path.join(model_dir, f'{model_name.lower()}_losses.csv'), index=False)

        # Save predictions to CSV
        pred_df = pd.DataFrame(
            {
                'actual': actual,
                'predicted': predicted,
                'residual': actual - predicted,
                'abs_error': np.abs(actual - predicted),
                'percentage_error': np.abs((actual - predicted) / actual) * 100,
            }
        )
        pred_df.to_csv(os.path.join(model_dir, f'{model_name.lower()}_predictions.csv'), index=False)

    return comprehensive_results
