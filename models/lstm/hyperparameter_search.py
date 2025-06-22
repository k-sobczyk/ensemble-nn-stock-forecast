import numpy as np
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data_processing import create_sequences_by_company
from .model import ModelLSTM, StockDatasetLSTM
from .training import train_model


def objective(
    trial: optuna.Trial,
    X_train_scaled,
    y_train_scaled,
    X_test_scaled,
    y_test_scaled,
    train_company_ids,
    test_company_ids,
    num_features,
    num_companies,
    device,
) -> float:
    """Optuna objective function for LSTM hyperparameter optimization."""
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'RMSprop'])
    sequence_length = trial.suggest_int('sequence_length', 2, 8)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    embedding_dim = trial.suggest_categorical('embedding_dim', [4, 8, 16, 32])
    min_samples = trial.suggest_int('min_samples_per_company', 3, 8)

    # Fixed training parameters
    epochs = 75
    patience = 10
    lr_scheduler_patience = 4
    lr_scheduler_factor = 0.25

    # Create sequences respecting company boundaries
    X_train_seq, y_train_seq, train_company_seq = create_sequences_by_company(
        X_train_scaled, y_train_scaled, train_company_ids, sequence_length, min_samples_per_company=min_samples
    )
    X_test_seq, y_test_seq, test_company_seq = create_sequences_by_company(
        X_test_scaled, y_test_scaled, test_company_ids, sequence_length, min_samples_per_company=min_samples
    )

    # Check if we have enough sequences
    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print(f'Trial {trial.number}: Not enough data for sequence_length {sequence_length}. Returning inf.')
        return float('inf')

    if X_train_seq.shape[0] < 30:  # Arbitrary threshold for minimum training samples
        print(f'Trial {trial.number}: Only {X_train_seq.shape[0]} training sequences created. Returning inf.')
        return float('inf')

    # Create datasets and dataloaders
    train_dataset = StockDatasetLSTM(X_train_seq, y_train_seq, train_company_seq)
    test_dataset = StockDatasetLSTM(X_test_seq, y_test_seq, test_company_seq)

    pin_memory = True if device.type == 'cuda' else False
    num_workers = 2 if device.type == 'cuda' else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    # Initialize model
    model = ModelLSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        num_companies=num_companies,
        embedding_dim=embedding_dim,
    )

    # Loss function, optimizer and scheduler
    criterion = nn.MSELoss()

    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=False, min_lr=1e-8
    )

    # Train and evaluate
    _, _, best_test_loss = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=epochs,
        patience=patience,
        device=device,
        trial=trial,
        verbose=False,
    )

    if not np.isfinite(best_test_loss):
        print(f'Warning: Trial {trial.number} resulted in non-finite loss: {best_test_loss}')
        return float('inf')

    return best_test_loss


def run_hyperparameter_search(
    X_train_scaled,
    y_train_scaled,
    X_test_scaled,
    y_test_scaled,
    train_company_ids,
    test_company_ids,
    num_features,
    num_companies,
    device,
    n_trials=50,
    timeout=3600,
):
    """Run Optuna hyperparameter search and return the best parameters."""
    objective_wrapper = lambda trial: objective(
        trial,
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        y_test_scaled,
        train_company_ids,
        test_company_ids,
        num_features,
        num_companies,
        device,
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=1)

    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

    try:
        study.optimize(objective_wrapper, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    except KeyboardInterrupt:
        print('Optuna optimization stopped manually.')
    except Exception as e:
        print(f'An error occurred during Optuna optimization: {e}')

    print('\n--- Optuna Optimization Finished ---')

    if not study.trials:
        print('No Optuna trials completed.')
        return None

    try:
        best_trial = study.best_trial
        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trial:')
        print(f'  Value (Best Test Loss - Scaled MSE): {best_trial.value:.6f}')
        print('  Best Parameters:')
        for key, value in best_trial.params.items():
            print(f'    {key}: {value}')
        return best_trial.params
    except ValueError:
        print('Optuna study finished, but no successful trials were completed.')
        return None
