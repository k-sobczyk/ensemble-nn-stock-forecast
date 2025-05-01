import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    def __init__(self, X: np.ndarray, y: np.ndarray, company_ids: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        if company_ids is not None:
            self.company_ids = torch.tensor(company_ids, dtype=torch.long)

        self.has_company_ids = company_ids is not None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        if self.has_company_ids:
            return self.X[idx], self.y[idx], self.company_ids[idx]
        else:
            return self.X[idx], self.y[idx]


class ModelLSTM(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_layers: int, dropout_rate: float,
                 use_batch_norm: bool = True, num_companies: int = 0, embedding_dim: int = 8):
        super(ModelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_company_embeddings = num_companies > 0

        if self.use_company_embeddings:
            self.company_embedding = nn.Embedding(num_companies, embedding_dim)
            lstm_input_size = num_features + embedding_dim
        else:
            lstm_input_size = num_features

        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, company_ids=None):
        if self.use_company_embeddings and company_ids is not None:
            embeds = self.company_embedding(company_ids)  # [batch_size, embedding_dim]
            seq_len = x.size(1)
            embeds = embeds.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embedding_dim]

            x = torch.cat([x, embeds], dim=2)  # [batch_size, seq_len, features+embedding_dim]

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output

        if self.use_batch_norm:
            out = self.bn(out)

        out = self.dropout(out)
        out = self.fc(out)
        return out


def load_and_split_data(file_path: str, date_column: str, test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Loads data, splits into train/test based on years."""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=[date_column])

    test_mask = df[date_column].dt.year.isin(test_years)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    total_samples = len(df)
    train_samples = len(train_df)
    test_samples = len(test_df)

    test_ratio = test_samples / total_samples * 100 if total_samples > 0 else 0

    print(f"Data Split: Total: {total_samples}, Train: {train_samples}, Test: {test_samples} ({test_ratio:.2f}%)")

    if 'ticker' in df.columns:
        print(f"Number of unique companies (tickers): {df['ticker'].nunique()}")
        print(f"Train companies: {train_df['ticker'].nunique()}, Test companies: {test_df['ticker'].nunique()}")

    return train_df, test_df, test_ratio


def prepare_base_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str, 
                      ticker_column: str = 'ticker') -> tuple:
    """Scales features and target, handles ticker information."""
    has_ticker = ticker_column in train_df.columns

    # Extract tickers before dropping
    if has_ticker:
        train_tickers = train_df[ticker_column].values
        test_tickers = test_df[ticker_column].values

        # Encode company tickers
        ticker_encoder = LabelEncoder()
        train_ticker_encoded = ticker_encoder.fit_transform(train_tickers)
        test_ticker_encoded = ticker_encoder.transform(test_tickers)

        print(f"Encoded {len(ticker_encoder.classes_)} unique companies")
        num_companies = len(ticker_encoder.classes_)
    else:
        train_ticker_encoded = None
        test_ticker_encoded = None
        num_companies = 0

    numeric_cols = [col for col in train_df.columns
                   if col != target_column and col != ticker_column and
                   (pd.api.types.is_numeric_dtype(train_df[col]) or col.startswith('sector_'))]

    print(f"Using {len(numeric_cols)} numeric features")

    X_train_raw = train_df[numeric_cols].values
    y_train_raw = train_df[target_column].values.reshape(-1, 1)
    X_test_raw = test_df[numeric_cols].values
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

    # Create a mapping between columns and their indices for feature importance analysis
    feature_indices = {col: i for i, col in enumerate(numeric_cols)}

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            y_train_raw.flatten(), scaler_X, scaler_y, num_features, 
            train_ticker_encoded, test_ticker_encoded, num_companies, feature_indices)


def create_sequences_by_company(X_scaled: np.ndarray, y_scaled: np.ndarray, 
                             company_ids: np.ndarray = None, sequence_length: int = 2,
                             min_samples_per_company: int = 4) -> tuple:
    """Creates sequences for LSTM input, respecting company boundaries with handling for sparse data.
    
    Args:
        X_scaled: Scaled feature matrix
        y_scaled: Scaled target values
        company_ids: Company identifiers for each sample
        sequence_length: Number of timesteps to include in each sequence (default: 2 for quarterly data)
        min_samples_per_company: Minimum number of samples required per company (default: 4)
    
    Returns:
        Tuple of sequences and corresponding targets (and company ids if provided)
    """
    X_seq, y_seq, company_seq = [], [], []
    
    if company_ids is not None:
        # Group by company to create sequences
        unique_companies = np.unique(company_ids)
        for company in unique_companies:
            # Get indices for this company
            company_mask = company_ids == company
            company_X = X_scaled[company_mask]
            company_y = y_scaled[company_mask]
            
            # Skip companies with too few observations
            if len(company_X) < min_samples_per_company:
                print(f"Company ID {company} has only {len(company_X)} samples, skipping (min required: {min_samples_per_company})")
                continue
                
            # Use shorter sequence length if needed but at least 2
            actual_seq_length = min(sequence_length, len(company_X) - 1)
            actual_seq_length = max(actual_seq_length, 2)  # At least 2 for minimal sequence
            
            if actual_seq_length < sequence_length:
                print(f"Reducing sequence length to {actual_seq_length} for company ID {company} (original: {sequence_length})")
            
            # Create sequences for this company
            for i in range(len(company_X) - actual_seq_length):
                X_seq.append(company_X[i:i + actual_seq_length])
                y_seq.append(company_y[i + actual_seq_length])
                company_seq.append(company)
    else:
        # If no company info, create regular sequences with minimal length check
        actual_seq_length = min(sequence_length, len(X_scaled) - 1)
        actual_seq_length = max(actual_seq_length, 2)
        
        for i in range(len(X_scaled) - actual_seq_length):
            X_seq.append(X_scaled[i:i + actual_seq_length])
            y_seq.append(y_scaled[i + actual_seq_length])
    
    if not X_seq:
        print(f"Warning: No sequences created with sequence_length={sequence_length}. Consider reducing sequence length.")
        num_features = X_scaled.shape[1] if X_scaled.ndim > 1 else 1
        if company_ids is not None:
            return np.empty((0, sequence_length, num_features)), np.empty((0, 1)), np.empty((0,))
        else:
            return np.empty((0, sequence_length, num_features)), np.empty((0, 1))
    
    # Pad sequences to the same length if they have different lengths
    max_seq_len = max(len(seq) for seq in X_seq)
    padded_X_seq = []
    
    for seq in X_seq:
        if len(seq) < max_seq_len:
            # Pad the sequence with zeros
            pad_size = max_seq_len - len(seq)
            padded_seq = np.vstack([np.zeros((pad_size, seq.shape[1])), seq])
            padded_X_seq.append(padded_seq)
        else:
            padded_X_seq.append(seq)
    
    if company_ids is not None:
        return np.array(padded_X_seq), np.array(y_seq), np.array(company_seq)
    else:
        return np.array(padded_X_seq), np.array(y_seq)


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
        for batch in train_loader:
            # Handle batches with or without company_ids
            if len(batch) == 3:
                inputs, targets, company_ids = [item.to(device) for item in batch]
                outputs = model(inputs, company_ids)
            else:
                inputs, targets = [item.to(device) for item in batch]
                outputs = model(inputs)

            optimizer.zero_grad()
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
            for batch in test_loader:
                # Handle batches with or without company_ids
                if len(batch) == 3:
                    inputs, targets, company_ids = [item.to(device) for item in batch]
                    outputs = model(inputs, company_ids)
                else:
                    inputs, targets = [item.to(device) for item in batch]
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
    company_ids_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle batches with or without company_ids
            if len(batch) == 3:
                inputs, targets, company_ids = [item.to(device) for item in batch]
                outputs = model(inputs, company_ids)
                company_ids_list.extend(company_ids.cpu().numpy())
            else:
                inputs, targets = [item.to(device) for item in batch]
                outputs = model(inputs)
                
            predictions_scaled.extend(outputs.cpu().numpy())
            actuals_scaled.extend(targets.cpu().numpy())

    predictions_scaled = np.array(predictions_scaled)
    actuals_scaled = np.array(actuals_scaled)
    
    if len(company_ids_list) > 0:
        company_ids_array = np.array(company_ids_list)
    else:
        company_ids_array = None

    if predictions_scaled.ndim == 1:
        predictions_scaled = predictions_scaled.reshape(-1, 1)
    if actuals_scaled.ndim == 1:
        actuals_scaled = actuals_scaled.reshape(-1, 1)

    if predictions_scaled.shape[0] == 0 or actuals_scaled.shape[0] == 0:
        print("Warning: Evaluating model with empty data.")
        return np.array([]), np.array([]), company_ids_array

    predictions = scaler_y.inverse_transform(predictions_scaled)
    actuals = scaler_y.inverse_transform(actuals_scaled)

    return predictions.flatten(), actuals.flatten(), company_ids_array


def objective(trial: optuna.Trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
              train_company_ids, test_company_ids, num_features, num_companies, device) -> float:
    """Optuna objective function for LSTM with company embeddings."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop"])

    # Use shorter sequence lengths for quarterly data
    sequence_length = trial.suggest_int("sequence_length", 2, 8)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    
    # Only use embeddings if we have company ids
    use_company_embeddings = train_company_ids is not None
    if use_company_embeddings:
        embedding_dim = trial.suggest_categorical("embedding_dim", [4, 8, 16, 32])
        # Minimum samples required per company (avoid companies with too few data points)
        min_samples = trial.suggest_int("min_samples_per_company", 3, 8)
    else:
        embedding_dim = 0
        min_samples = 4  # Default

    epochs = 75
    patience = 10
    lr_scheduler_patience = 4
    lr_scheduler_factor = 0.25

    # Create sequences respecting company boundaries
    if use_company_embeddings:
        X_train_seq, y_train_seq, train_company_seq = create_sequences_by_company(
            X_train_scaled, y_train_scaled, train_company_ids, 
            sequence_length, min_samples_per_company=min_samples)
        X_test_seq, y_test_seq, test_company_seq = create_sequences_by_company(
            X_test_scaled, y_test_scaled, test_company_ids, 
            sequence_length, min_samples_per_company=min_samples)
    else:
        X_train_seq, y_train_seq = create_sequences_by_company(
            X_train_scaled, y_train_scaled, sequence_length=sequence_length)
        X_test_seq, y_test_seq = create_sequences_by_company(
            X_test_scaled, y_test_scaled, sequence_length=sequence_length)
        train_company_seq = None
        test_company_seq = None

    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print(f"Trial {trial.number}: Not enough data for sequence_length {sequence_length}. Returning inf.")
        return float('inf')
        
    # Check if we have enough sequences
    if X_train_seq.shape[0] < 30:  # Arbitrary threshold for minimum training samples
        print(f"Trial {trial.number}: Only {X_train_seq.shape[0]} training sequences created. Returning inf.")
        return float('inf')

    # Create datasets with or without company ids
    if use_company_embeddings:
        train_dataset = StockDatasetLSTM(X_train_seq, y_train_seq, train_company_seq)
        test_dataset = StockDatasetLSTM(X_test_seq, y_test_seq, test_company_seq)
    else:
        train_dataset = StockDatasetLSTM(X_train_seq, y_train_seq)
        test_dataset = StockDatasetLSTM(X_test_seq, y_test_seq)

    pin_memory = True if device.type == 'cuda' else False
    num_workers = 2 if device.type == 'cuda' else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    # Initialize model with or without company embeddings
    if use_company_embeddings:
        model = ModelLSTM(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            num_companies=num_companies,
            embedding_dim=embedding_dim
        )
    else:
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
    DATA_FILE = r'C:\Users\ksobc\PycharmProjects\ensemble-nn-stock-forecast\src\models\model_with_features.csv'
    DATE_COLUMN = 'end_of_period'
    TARGET_COLUMN = 'target'
    TICKER_COLUMN = 'ticker'  # Column with company identifiers
    TEST_YEARS = [2021, 2022]
    
    # Parameters for quarterly stock data
    DEFAULT_SEQ_LENGTH = 2  # Default sequence length for quarterly data
    MIN_SAMPLES_PER_COMPANY = 4  # Minimum number of samples needed per company

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
    
    # Check the number of data points per company
    if TICKER_COLUMN in train_df.columns:
        print("\n--- Samples per company (training set) ---")
        company_counts = train_df[TICKER_COLUMN].value_counts()
        print(f"Min samples: {company_counts.min()}, Max samples: {company_counts.max()}, Median: {company_counts.median()}")
        print(f"Companies with less than {MIN_SAMPLES_PER_COMPANY} samples: {sum(company_counts < MIN_SAMPLES_PER_COMPANY)}")
        
        # Display sample sizes for a few companies
        print("\nSample counts for first 10 companies:")
        for company, count in company_counts.head(10).items():
            print(f"{company}: {count} samples")
    
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, \
    y_train_raw, scaler_X, scaler_y, num_features, \
    train_ticker_encoded, test_ticker_encoded, \
    num_companies, feature_indices = prepare_base_data(
        train_df, test_df, TARGET_COLUMN, TICKER_COLUMN
    )

    print(f"\n--- Starting Optuna Hyperparameter Optimization ({N_TRIALS} trials) ---")

    objective_wrapper = lambda trial: objective(
        trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
        train_ticker_encoded, test_ticker_encoded, num_features, num_companies, device
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
    
    # Get min_samples parameter from best params if available
    min_samples_per_company = best_params.get('min_samples_per_company', MIN_SAMPLES_PER_COMPANY)
    
    # Create sequences for final model with best parameters
    use_company_embeddings = train_ticker_encoded is not None
    if use_company_embeddings:
        X_train_final, y_train_final, train_company_final = create_sequences_by_company(
            X_train_scaled, y_train_scaled, train_ticker_encoded, 
            best_sequence_length, min_samples_per_company=min_samples_per_company)
        X_test_final, y_test_final, test_company_final = create_sequences_by_company(
            X_test_scaled, y_test_scaled, test_ticker_encoded, 
            best_sequence_length, min_samples_per_company=min_samples_per_company)
    else:
        X_train_final, y_train_final = create_sequences_by_company(
            X_train_scaled, y_train_scaled, sequence_length=best_sequence_length)
        X_test_final, y_test_final = create_sequences_by_company(
            X_test_scaled, y_test_scaled, sequence_length=best_sequence_length)
        train_company_final = None
        test_company_final = None

    # Print sequence shape information
    print(f"\nCreated {X_train_final.shape[0]} training sequences and {X_test_final.shape[0]} test sequences")
    print(f"Sequence shape: {X_train_final.shape}")

    if X_train_final.shape[0] == 0 or X_test_final.shape[0] == 0:
        print("Error: Cannot retrain model as the best sequence length resulted in empty data.")
        return

    # Create datasets for final model
    if use_company_embeddings:
        final_train_dataset = StockDatasetLSTM(X_train_final, y_train_final, train_company_final)
        final_test_dataset = StockDatasetLSTM(X_test_final, y_test_final, test_company_final)
    else:
        final_train_dataset = StockDatasetLSTM(X_train_final, y_train_final)
        final_test_dataset = StockDatasetLSTM(X_test_final, y_test_final)

    pin_memory = True if device.type == 'cuda' else False
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_params['batch_size'], shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    final_test_loader = DataLoader(final_test_dataset, batch_size=best_params['batch_size'], shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory)

    # Initialize final model with best parameters
    if use_company_embeddings:
        embedding_dim = best_params.get("embedding_dim", 8)
        final_model = ModelLSTM(
            num_features=num_features,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout_rate=best_params['dropout_rate'],
            use_batch_norm=best_params['use_batch_norm'],
            num_companies=num_companies,
            embedding_dim=embedding_dim
        )
    else:
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
    y_pred, y_true, company_ids = evaluate_model(final_trained_model, final_test_loader, scaler_y, device=device)

    if len(y_true) > 0 and len(y_pred) > 0:
        print_metrics(y_true, y_pred, y_train_raw)

        # Create results DataFrame with company information if available
        results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        
        # Add company information if available
        if company_ids is not None and len(company_ids) == len(y_true):
            ticker_encoder = LabelEncoder()
            ticker_encoder.fit(train_df[TICKER_COLUMN])
            results_df['Company'] = ticker_encoder.inverse_transform(company_ids)
            
            # Calculate metrics per company
            print("\n--- Metrics by Company ---")
            for company in results_df['Company'].unique():
                company_mask = results_df['Company'] == company
                company_true = results_df.loc[company_mask, 'Actual'].values
                company_pred = results_df.loc[company_mask, 'Predicted'].values
                
                if len(company_true) > 5:  # Reduced threshold for quarterly data
                    print(f"\nCompany: {company} (samples: {len(company_true)})")
                    print_metrics(company_true, company_pred, y_train_raw)
        
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
