from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.model.individual.bi_lstm import train_bi_lstm_model
from src.model.individual.cnn import train_cnn_model
from src.model.individual.gru import train_gru_model

# Import individual models
from src.model.individual.lstm import train_lstm_model
from src.model.individual.model_utils import prepare_data


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods."""

    def __init__(self, models_config=None):
        self.models_config = models_config or {'lstm': True, 'gru': True, 'bi_lstm': True, 'cnn': True}
        self.trained_models = {}
        self.model_predictions = {}

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        """Train the ensemble on training data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions using the ensemble."""
        pass

    def evaluate(self, y_true, y_pred, scaler_y=None):
        """Evaluate ensemble performance."""
        if scaler_y is not None:
            y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mse': mse}


class ModelTrainer:
    """Utility class for training individual models."""

    @staticmethod
    def train_individual_models(
        X_train, y_train, X_val, y_val, input_size, sequence_length=None, models_config=None, epochs=50
    ):
        """Train all specified individual models."""
        models_config = models_config or {'lstm': True, 'gru': True, 'bi_lstm': True, 'cnn': True}

        trained_models = {}

        if models_config.get('lstm', True):
            print('Training LSTM...')
            lstm_model, _, _ = train_lstm_model(X_train, y_train, X_val, y_val, input_size, epochs=epochs)
            trained_models['lstm'] = lstm_model

        if models_config.get('gru', True):
            print('Training GRU...')
            gru_model, _, _ = train_gru_model(X_train, y_train, X_val, y_val, input_size, epochs=epochs)
            trained_models['gru'] = gru_model

        if models_config.get('bi_lstm', True):
            print('Training Bi-LSTM...')
            bilstm_model, _, _ = train_bi_lstm_model(X_train, y_train, X_val, y_val, input_size, epochs=epochs)
            trained_models['bi_lstm'] = bilstm_model

        if models_config.get('cnn', True) and sequence_length is not None:
            print('Training CNN...')
            cnn_model, _, _ = train_cnn_model(
                X_train, y_train, X_val, y_val, input_size, sequence_length, epochs=epochs
            )
            trained_models['cnn'] = cnn_model

        return trained_models

    @staticmethod
    def get_model_predictions(models_dict, X_data, model_type='test'):
        """Get predictions from all trained models."""
        predictions = {}

        for model_name, model in models_dict.items():
            model.eval()
            # Get the device of the model
            device = next(model.parameters()).device

            with torch.no_grad():
                if model_name == 'cnn':
                    # CNN needs different data format
                    test_dataset = StockDatasetCNN(X_data)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

                    model_preds = []
                    for batch_X in test_loader:
                        # Move batch to same device as model
                        batch_X = batch_X.to(device)
                        batch_pred = model(batch_X).cpu().numpy()  # Move to CPU for numpy conversion
                        model_preds.extend(batch_pred)
                    predictions[model_name] = np.array(model_preds).flatten()
                else:
                    # RNN-based models
                    # Move tensor to same device as model
                    X_tensor = torch.FloatTensor(X_data).to(device)
                    pred = model(X_tensor).cpu().numpy().flatten()  # Move to CPU for numpy conversion
                    predictions[model_name] = pred

        return predictions


class StockDatasetCNN:
    """Simple dataset class for CNN predictions."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Transpose for CNN: (features, time_steps)
        return torch.FloatTensor(self.sequences[idx]).transpose(0, 1)


def create_cross_validation_folds(X_train, y_train, n_folds=5):
    """Create cross-validation folds for ensemble training."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

    return folds


def prepare_ensemble_data(
    df, sequence_length=None, test_start_year=2021, auto_sequence_length=True, split_validation=True
):
    """Prepare data for ensemble training with optional validation split."""
    # Use existing prepare_data function
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols = prepare_data(
        df,
        sequence_length=sequence_length,
        test_start_year=test_start_year,
        auto_sequence_length=auto_sequence_length,
        model_type='rnn',
    )

    if split_validation:
        # Split training data into train/validation
        split_idx = int(0.8 * len(X_train))
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y, feature_cols
    else:
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols
