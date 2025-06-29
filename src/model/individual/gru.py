import warnings

import optuna
import pandas as pd
import torch
import torch.nn as nn

from src.model.individual.config import EARLY_STOPPAGE, EPOCHS
from src.model.individual.model_utils import EarlyStopping, create_data_loaders, evaluate_model, prepare_data

warnings.filterwarnings('ignore')


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.4):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        # Batch normalization after GRU
        self.batch_norm_gru = nn.BatchNorm1d(hidden_size)

        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Get device from input tensor
        device = x.device
        batch_size = x.size(0)

        # Initialize hidden state on the same device as input
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # GRU forward pass
        gru_out, _ = self.gru(x, h0)

        # Use the last time step output
        last_output = gru_out[:, -1, :]

        # Apply batch normalization
        last_output = self.batch_norm_gru(last_output)

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output


def train_gru_model(
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
    max_grad_norm=1.0,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5,
    hidden_size=64,
    num_layers=2,
    dropout=0.4,
    verbose=True,
):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Using device: {device}')

        if device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    # Create datasets and dataloaders using common utility
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size, model_type='rnn')

    # Initialize model and move to device
    model = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=verbose, min_lr=1e-6
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.0001, restore_best_weights=True)

    # Training loop
    train_losses = []
    test_losses = []

    if verbose:
        print('\nTraining GRU model...')
        print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        print(f'Early stopping patience: {early_stopping_patience} epochs')
        print(f'Learning rate: {learning_rate}, Weight decay: {weight_decay}')
        print(f'Hidden size: {hidden_size}, Layers: {num_layers}, Dropout: {dropout}')
        print(f'Max gradient norm: {max_grad_norm}')

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

            # Gradient clipping
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

        # Step the scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(
                f'Epoch [{epoch + 1:3d}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.2e}'
            )

            # GPU memory usage (if using CUDA)
            if device.type == 'cuda' and (epoch + 1) % 10 == 0:
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB')

        # Early stopping check
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            if verbose:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                print(f'Best validation loss: {early_stopping.best_loss:.4f}')
            break

    # Restore best model weights
    early_stopping.restore_best_weights_to_model(model)
    if verbose:
        print(f'Restored best model weights (validation loss: {early_stopping.best_loss:.4f})')

    return model, train_losses, test_losses, early_stopping.best_loss


def objective(trial, X_train, y_train, X_test, y_test, input_size):
    """Optuna objective function for GRU hyperparameter optimization."""
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.6, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0, step=0.1)
    lr_scheduler_patience = trial.suggest_int('lr_scheduler_patience', 3, 10)
    lr_scheduler_factor = trial.suggest_float('lr_scheduler_factor', 0.3, 0.8, step=0.1)

    try:
        # Train model with suggested hyperparameters
        model, train_losses, test_losses, best_val_loss = train_gru_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            input_size=input_size,
            epochs=50,  # Reduced for optimization speed
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=15,  # Reduced for optimization speed
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            verbose=False,  # Suppress output during optimization
        )

        return best_val_loss

    except Exception as e:
        print(f'Trial failed with error: {e}')
        return float('inf')


def optimize_hyperparameters(X_train, y_train, X_test, y_test, input_size, n_trials=50):
    """Run Optuna hyperparameter optimization for GRU model."""
    print('=' * 80)
    print('🔍 STARTING GRU HYPERPARAMETER OPTIMIZATION')
    print('=' * 80)
    print(f'Number of trials: {n_trials}')
    print(f'Training samples: {X_train.shape[0]}')
    print(f'Test samples: {X_test.shape[0]}')
    print(f'Input features: {input_size}')
    print('-' * 80)

    # Create study
    study = optuna.create_study(direction='minimize', study_name='gru_optimization')

    # Define objective function with data
    def objective_with_data(trial):
        return objective(trial, X_train, y_train, X_test, y_test, input_size)

    # Optimize
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)

    print('\n' + '=' * 80)
    print('✅ OPTIMIZATION COMPLETED')
    print('=' * 80)

    # Print best results
    best_trial = study.best_trial
    print(f'Best validation loss: {best_trial.value:.4f}')
    print(f'Best trial number: {best_trial.number}')

    print('\n🏆 BEST HYPERPARAMETERS:')
    print('-' * 40)
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f'{key:20s}: {value:.6f}')
        else:
            print(f'{key:20s}: {value}')

    # Generate config code
    print('\n📋 CONFIG CODE (add to config.py):')
    print('-' * 40)
    params = best_trial.params
    print(f'# Optimized GRU parameters (val_loss: {best_trial.value:.4f})')
    print(f'GRU_HIDDEN_SIZE = {params["hidden_size"]}')
    print(f'GRU_NUM_LAYERS = {params["num_layers"]}')
    print(f'GRU_DROPOUT = {params["dropout"]}')
    print(f'GRU_LEARNING_RATE = {params["learning_rate"]:.6f}')
    print(f'GRU_WEIGHT_DECAY = {params["weight_decay"]:.6f}')
    print(f'GRU_BATCH_SIZE = {params["batch_size"]}')
    print(f'GRU_MAX_GRAD_NORM = {params["max_grad_norm"]}')
    print(f'GRU_LR_SCHEDULER_PATIENCE = {params["lr_scheduler_patience"]}')
    print(f'GRU_LR_SCHEDULER_FACTOR = {params["lr_scheduler_factor"]}')

    return study.best_params, best_trial.value


def main(
    sequence_length=None,
    auto_sequence_length=True,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPAGE,
    optimize=False,
    n_trials=50,
):
    print('=' * 60)
    print('GRU Stock Price Prediction - GPW Dataset')
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

    input_size = len(feature_cols)
    print(f'\nModel input size: {input_size} features')

    if optimize:
        # Run hyperparameter optimization
        best_params, best_loss = optimize_hyperparameters(X_train, y_train, X_test, y_test, input_size, n_trials)

        # Train final model with best parameters
        print('\n🚀 Training final model with optimized parameters...')
        model, train_losses, test_losses, _ = train_gru_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            input_size=input_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            **best_params,
            verbose=True,
        )
    else:
        # Train model with default parameters
        model, train_losses, test_losses, _ = train_gru_model(
            X_train, y_train, X_test, y_test, input_size, epochs=epochs, early_stopping_patience=early_stopping_patience
        )

    # Evaluate model using common utility
    results = evaluate_model(model, X_test, y_test, scaler_y, model_type='rnn', model_name='GRU Model')

    return model, results


if __name__ == '__main__':
    # Set optimize=True to run hyperparameter optimization
    model, results = main(optimize=True, n_trials=50)
