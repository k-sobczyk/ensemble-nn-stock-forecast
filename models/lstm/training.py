import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import time
import optuna
from sklearn.preprocessing import StandardScaler


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, scheduler: ReduceLROnPlateau, epochs: int = 100,
                patience: int = 10, device: torch.device = torch.device("cpu"),
                trial: optuna.Trial = None, verbose: bool = True) -> tuple[pd.DataFrame, nn.Module, float]:
    """Train LSTM model with early stopping and learning rate scheduling."""
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
            inputs, targets, company_ids = [item.to(device) for item in batch]
            outputs = model(inputs, company_ids)

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
                inputs, targets, company_ids = [item.to(device) for item in batch]
                outputs = model(inputs, company_ids)

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
                   device: torch.device = torch.device("cpu")) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate model on test data and return predictions, actual values, and company IDs."""
    model.eval()
    predictions_scaled = []
    actuals_scaled = []
    company_ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, company_ids = [item.to(device) for item in batch]
            outputs = model(inputs, company_ids)
            company_ids_list.extend(company_ids.cpu().numpy())

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
