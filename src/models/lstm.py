import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import warnings
import optuna
from metrics.metrics import print_metrics


warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, message="The verbose parameter is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Seems like `optuna.trial._trial.Trial` is deprecated")


class StockDatasetLSTM(Dataset):
    """Dataset class for LSTM, expects sequenced data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ModelLSTM(nn.Module):
    """LSTM Model with optional BatchNorm and Dropout."""
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout_rate: float, use_batch_norm: bool = True):
        super(ModelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        self.lstm = nn.LSTM(num_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        if self.use_batch_norm:
            out = self.bn(out)

        out = self.dropout(out)
        out = self.fc(out)
        return out


def load_and_split_data(file_path: str, date_column: str, test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Loads data, splits into train/test based on years."""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    test_mask = df[date_column].dt.year.isin(test_years)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    train_dates = train_df[date_column]
    test_dates = test_df[date_column]
    train_df = train_df.drop(columns=[date_column])
    test_df = test_df.drop(columns=[date_column])

    total_samples = len(df)
    train_samples = len(train_df)
    test_samples = len(test_df)

    test_ratio = test_samples / total_samples * 100 if total_samples > 0 else 0

    print(f"Data Split: Total: {total_samples}, Train: {train_samples}, Test: {test_samples} ({test_ratio:.2f}%)")

    return train_df, test_df, test_ratio


def prepare_base_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler, int]:
    """Scales features and target, returns unsequenced data."""
    X_train_raw = train_df.drop(columns=[target_column]).values
    y_train_raw = train_df[target_column].values.reshape(-1, 1)
    X_test_raw = test_df.drop(columns=[target_column]).values
    y_test_raw = test_df[target_column].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    num_features = X_train_scaled.shape[1]

    print(f"Number of features: {num_features}")
    print(f"Scaled Train Shapes: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"Scaled Test Shapes: X={X_test_scaled.shape}, y={y_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_train_raw.flatten(), scaler_X, scaler_y, num_features


def create_sequences(X_scaled: np.ndarray, y_scaled: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    if not X_seq:
        print(f"Warning: Data length ({len(X_scaled)}) is less than or equal to sequence length ({sequence_length}). No sequences created.")
        num_features = X_scaled.shape[1] if X_scaled.ndim > 1 else 1
        return np.empty((0, sequence_length, num_features)), np.empty((0, y_scaled.shape[1] if y_scaled.ndim > 1 else 1))

    return np.array(X_seq), np.array(y_seq)


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, scheduler: ReduceLROnPlateau, epochs: int = 100,
                patience: int = 10, device: torch.device = torch.device("cpu"),
                trial: optuna.Trial = None, verbose: bool = True) -> tuple[pd.DataFrame, nn.Module, float]:

    model.to(device)
    train_losses = []
    test_losses = []
    learning_rates = []
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * inputs.size(0)

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f'Epoch {epoch + 1:03d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f} | LR: {current_lr:.6f}')

        scheduler.step(epoch_test_loss)

        is_best = epoch_test_loss < best_test_loss
        if is_best:
            best_test_loss = epoch_test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if trial is not None:
            if np.isfinite(best_test_loss):
                trial.report(best_test_loss, epoch)
            else:
                trial.report(float('inf'), epoch)

            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                raise optuna.TrialPruned()

        if current_lr < 1e-7 and epoch > 10:
            if verbose:
                print(f"Stopping early due to very small learning rate at epoch {epoch + 1}")
            break

    if verbose:
        training_time = time.time() - start_time
        print(f"Training finished in {training_time:.2f} seconds")

    loss_df = pd.DataFrame({
        'Epoch': list(range(1, len(train_losses) + 1)),
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Learning Rate': learning_rates
    })

    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state was saved.")

    final_best_loss = best_test_loss if np.isfinite(best_test_loss) else float('inf')

    return loss_df, model, final_best_loss


def evaluate_model(model: nn.Module, test_loader: DataLoader, scaler_y: StandardScaler,
                   device: torch.device = torch.device("cpu")) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions_scaled = []
    actuals_scaled = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions_scaled.extend(outputs.cpu().numpy())
            actuals_scaled.extend(targets.cpu().numpy())

    predictions_scaled = np.array(predictions_scaled)
    actuals_scaled = np.array(actuals_scaled)

    if predictions_scaled.ndim == 1:
        predictions_scaled = predictions_scaled.reshape(-1, 1)
    if actuals_scaled.ndim == 1:
        actuals_scaled = actuals_scaled.reshape(-1, 1)

    if predictions_scaled.shape[0] == 0 or actuals_scaled.shape[0] == 0:
        print("Warning: Evaluating model with empty data.")
        return np.array([]), np.array([])

    predictions = scaler_y.inverse_transform(predictions_scaled)
    actuals = scaler_y.inverse_transform(actuals_scaled)

    return predictions.flatten(), actuals.flatten()


def objective(trial: optuna.Trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, num_features, device) -> float:
    """Optuna objective function for LSTM."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

    sequence_length = trial.suggest_int("sequence_length", 5, 60)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    epochs = 75
    patience = 10
    lr_scheduler_patience = 4
    lr_scheduler_factor = 0.25

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print(f"Trial {trial.number}: Not enough data for sequence_length {sequence_length}. Returning inf.")
        return float('inf')

    train_dataset = StockDatasetLSTM(X_train_seq, y_train_seq)
    test_dataset = StockDatasetLSTM(X_test_seq, y_test_seq)

    pin_memory = True if device.type == 'cuda' else False
    num_workers = 2 if device.type == 'cuda' else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    model = ModelLSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )

    criterion = nn.MSELoss()

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=False,
        min_lr=1e-8
    )

    _, _, best_test_loss = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        epochs=epochs, patience=patience, device=device, trial=trial, verbose=False
    )

    if not np.isfinite(best_test_loss):
        print(f"Warning: Trial {trial.number} resulted in non-finite loss: {best_test_loss}")
        return float('inf')

    return best_test_loss


def main():
    DATA_FILE = r'C:\Python\projects\ensemble-nn-stock-forecast\src\models\data\model_with_features.csv'
    DATE_COLUMN = 'end_of_period'
    TARGET_COLUMN = 'target'
    TEST_YEARS = [2021, 2022]

    N_TRIALS = 50
    OPTUNA_TIMEOUT = 3600

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        num_workers = 2
    else:
        num_workers = 0

    print(f"Using {num_workers} dataloader workers.")

    train_df, test_df, test_ratio = load_and_split_data(DATA_FILE, DATE_COLUMN, TEST_YEARS)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, \
    y_train_raw, scaler_X, scaler_y, num_features = prepare_base_data(
        train_df, test_df, TARGET_COLUMN
    )

    print(f"\n--- Starting Optuna Hyperparameter Optimization ({N_TRIALS} trials) ---")

    objective_wrapper = lambda trial: objective(
        trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, num_features, device
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=1)

    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

    try:
        study.optimize(objective_wrapper, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, gc_after_trial=True)
    except KeyboardInterrupt:
        print("Optuna optimization stopped manually.")
    except Exception as e:
        print(f"An error occurred during Optuna optimization: {e}")

    print("\n--- Optuna Optimization Finished ---")

    if not study.trials:
         print("No Optuna trials completed.")
         return

    try:
        best_trial = study.best_trial
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        print(f"  Value (Best Test Loss - Scaled MSE): {best_trial.value:.6f}")
        print("  Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("Optuna study finished, but no successful trials were completed.")
        return

    print("\n--- Retraining model with best parameters ---")
    best_params = best_trial.params

    best_sequence_length = best_params['sequence_length']
    print(f"Using best sequence length: {best_sequence_length}")
    X_train_final, y_train_final = create_sequences(X_train_scaled, y_train_scaled, best_sequence_length)
    X_test_final, y_test_final = create_sequences(X_test_scaled, y_test_scaled, best_sequence_length)

    if X_train_final.shape[0] == 0 or X_test_final.shape[0] == 0:
        print("Error: Cannot retrain model as the best sequence length resulted in empty data.")
        return

    final_train_dataset = StockDatasetLSTM(X_train_final, y_train_final)
    final_test_dataset = StockDatasetLSTM(X_test_final, y_test_final)

    pin_memory = True if device.type == 'cuda' else False
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params['batch_size'], shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    final_test_loader = DataLoader(final_test_dataset, batch_size=best_params['batch_size'], shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory)

    final_model = ModelLSTM(
        num_features=num_features,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        use_batch_norm=best_params['use_batch_norm']
    )

    criterion = nn.MSELoss()

    optimizer_name = best_params['optimizer']
    print(f"Using optimizer: {optimizer_name}")
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.RMSprop(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    final_epochs = 200
    final_patience = 25
    final_lr_patience = 10
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=final_lr_patience, verbose=True, min_lr=1e-8)

    loss_df, final_trained_model, _ = train_model(
        final_model, final_train_loader, final_test_loader, criterion, optimizer, scheduler,
        epochs=final_epochs, patience=final_patience, device=device, verbose=True
    )

    print("\n--- Evaluating final best model on Test Set ---")
    y_pred, y_true = evaluate_model(final_trained_model, final_test_loader, scaler_y, device=device)

    if len(y_true) > 0 and len(y_pred) > 0:
        print_metrics(y_true, y_pred, y_train_raw)

        results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        results_df.to_csv("lstm_best_model_predictions.csv", index=False, float_format='%.4f')
        print("\nBest LSTM model predictions saved to lstm_best_model_predictions.csv")

        loss_df.to_csv("lstm_best_model_training_history.csv", index=False, float_format='%.6f')
        print("Best LSTM model training history saved to lstm_best_model_training_history.csv")

        model_save_path = "lstm_best_model.pth"
        torch.save(final_trained_model.state_dict(), model_save_path)
        print(f"Best LSTM model state dict saved to {model_save_path}")

        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv("lstm_best_model_hyperparameters.csv", index=False)
        print("Best LSTM model hyperparameters saved to lstm_best_model_hyperparameters.csv")
    else:
        print("Evaluation failed: No predictions generated.")


if __name__ == "__main__":
    main()
