import numpy as np
from sklearn.metrics import mean_absolute_error


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Calculates Mean Absolute Scaled Error."""
    y_true, y_pred, y_train = np.array(y_true), np.array(y_pred), np.array(y_train)
    if len(y_train) < 2:
        print('Warning: MASE calculation requires at least 2 training samples for naive forecast.')
        return np.nan

    mae = mean_absolute_error(y_true, y_pred)
    naive_forecast_error = np.mean(np.abs(y_train[1:] - y_train[:-1]))

    if naive_forecast_error == 0:
        print('Warning: Naive forecast error on training data is zero. MASE is undefined or infinite.')
        return np.inf if mae > 0 else 0.0

    return mae / naive_forecast_error
