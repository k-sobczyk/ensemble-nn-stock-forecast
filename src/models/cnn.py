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


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ModelCNN(nn.Module):
    def __init__(self, num_features: int, num_filters1: int, num_filters2: int, fc1_neurons: int, dropout_rate: float):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters1, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(num_filters1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(num_filters2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.flattened_size = num_filters2 * num_features

        self.fc1 = nn.Linear(self.flattened_size, fc1_neurons)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_neurons, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


def load_and_split_data(file_path: str, date_column: str, test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    test_mask = df[date_column].dt.year.isin(test_years)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    train_df = train_df.drop(columns=[date_column])
    test_df = test_df.drop(columns=[date_column])

    total_samples = len(df)
    train_samples = len(train_df)
    test_samples = len(test_df)

    test_ratio = test_samples / total_samples * 100 if total_samples > 0 else 0

    print(f"Data Split: Total: {total_samples}, Train: {train_samples}, Test: {test_samples} ({test_ratio:.2f}%)")

    return train_df, test_df, test_ratio


def prepare_data_for_cnn(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
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
    X_train = X_train_scaled.reshape(-1, 1, num_features, 1)
    X_test = X_test_scaled.reshape(-1, 1, num_features, 1)

    return X_train, X_test, y_train_scaled, y_test_scaled, y_train_raw.flatten(), scaler_y, num_features


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
            trial.report(epoch_test_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if current_lr < 1e-7:
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

    return loss_df, model, best_test_loss


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

    predictions = scaler_y.inverse_transform(predictions_scaled)
    actuals = scaler_y.inverse_transform(actuals_scaled)

    return predictions.flatten(), actuals.flatten()


def objective(trial: optuna.Trial, X_train, y_train, X_test, y_test, num_features, device) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_filters1 = trial.suggest_categorical("num_filters1", [16, 32, 64])
    num_filters2 = trial.suggest_categorical("num_filters2", [32, 64, 128])
    fc1_neurons = trial.suggest_categorical("fc1_neurons", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

    epochs = 100
    patience = 15
    lr_scheduler_patience = 5
    lr_scheduler_factor = 0.2

    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    # Set pin_memory to True if using CUDA
    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin_memory)

    model = ModelCNN(
        num_features=num_features,
        num_filters1=num_filters1,
        num_filters2=num_filters2,
        fc1_neurons=fc1_neurons,
        dropout_rate=dropout_rate
    )
    criterion = nn.MSELoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=False,
        min_lr=1e-7
    )

    _, _, best_test_loss = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        epochs=epochs, patience=patience, device=device, trial=trial, verbose=False
    )

    return best_test_loss


def main():
    DATA_FILE = r'C:\Python\projects\ensemble-nn-stock-forecast\src\models\data\model_with_features.csv'
    DATE_COLUMN = 'end_of_period'
    TARGET_COLUMN = 'target'
    TEST_YEARS = [2021, 2022]

    N_TRIALS = 50
    OPTUNA_TIMEOUT = 3600

    # Set device to CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print more details if CUDA is available
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_df, test_df, test_ratio = load_and_split_data(DATA_FILE, DATE_COLUMN, TEST_YEARS)
    X_train, X_test, y_train, y_test, y_train_raw, scaler_y, num_features = prepare_data_for_cnn(
        train_df, test_df, TARGET_COLUMN
    )

    print(f"\n--- Starting Optuna Hyperparameter Optimization ({N_TRIALS} trials) ---")
    objective_wrapper = lambda trial: objective(trial, X_train, y_train, X_test, y_test, num_features, device)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

    study.optimize(objective_wrapper, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)

    print("\n--- Optuna Optimization Finished ---")

    if study.best_trial:
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value (Best Test Loss): {best_trial.value:.6f}")
        print("  Best Parameters: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print("\n--- Retraining model with best parameters ---")
        best_params = best_trial.params
        best_model = ModelCNN(
            num_features=num_features,
            num_filters1=best_params['num_filters1'],
            num_filters2=best_params['num_filters2'],
            fc1_neurons=best_params['fc1_neurons'],
            dropout_rate=best_params['dropout_rate']
        )
        criterion = nn.MSELoss()

        optimizer_name = best_params['optimizer']
        if optimizer_name == "Adam":
            optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        else:
            optimizer = optim.RMSprop(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

        print(f"Using optimizer: {optimizer_name}")

        # Set pin_memory to True if using CUDA
        pin_memory = True if device.type == 'cuda' else False

        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True,
                                 num_workers=0, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False,
                                num_workers=0, pin_memory=pin_memory)

        final_epochs = 200
        final_patience = 20
        final_lr_patience = 8
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=final_lr_patience, verbose=True, min_lr=1e-7)

        _, final_trained_model, _ = train_model(
            best_model, train_loader, test_loader, criterion, optimizer, scheduler,
            epochs=final_epochs, patience=final_patience, device=device
        )

        print("\n--- Evaluating final best model ---")
        y_pred, y_true = evaluate_model(final_trained_model, test_loader, scaler_y, device=device)
        print_metrics(y_true, y_pred, y_train_raw)

        results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        results_df.to_csv("cnn_best_model_predictions.csv", index=False, float_format='%.4f')
        print("\nBest model predictions saved to cnn_best_model_predictions.csv")


if __name__ == "__main__":
    main()
