import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


def calculate_symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (more robust for stock prices)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    non_zero_mask = denominator != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan

    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    denominator_filtered = denominator[non_zero_mask]

    return np.mean(np.abs(y_true_filtered - y_pred_filtered) / denominator_filtered) * 100


def calculate_mape_log_scale(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates MAPE on log scale (more appropriate for price data)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Ensure positive values for log transformation
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        print('Warning: Non-positive values found, using log1p transformation')
        y_true_log = np.log1p(y_true)
        y_pred_log = np.log1p(y_pred)
    else:
        y_true_log = np.log(y_true)
        y_pred_log = np.log(y_pred)

    non_zero_mask = y_true_log != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan

    y_true_filtered = y_true_log[non_zero_mask]
    y_pred_filtered = y_pred_log[non_zero_mask]

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


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates R-squared (coefficient of determination)."""
    return r2_score(y_true, y_pred)


def calculate_weighted_average_metrics(results_df):
    """Calculate sample-size weighted average metrics across multiple time series.

    This method weights each ticker's metrics by the number of test samples,
    giving more influence to tickers with more reliable (larger sample size) results.
    This is more statistically sound than simple averaging for time series evaluation.

    Args:
        results_df: DataFrame with columns ['rmse', 'mae', 'mape', 'mase', 'r2', 'n_test_samples']

    Returns:
        dict: Weighted average metrics
    """
    if 'n_test_samples' not in results_df.columns:
        print('Warning: n_test_samples not found, using simple mean')
        return {
            'rmse': results_df['rmse'].mean(),
            'mae': results_df['mae'].mean(),
            'mape': results_df['mape'].mean(),
            'smape': results_df['smape'].mean() if 'smape' in results_df.columns else np.nan,
            'mape_log': results_df['mape_log'].mean() if 'mape_log' in results_df.columns else np.nan,
            'mase': results_df['mase'].mean(),
            'r2': results_df['r2'].mean(),
        }

    total_samples = results_df['n_test_samples'].sum()
    weights = results_df['n_test_samples'] / total_samples

    return {
        'rmse': (results_df['rmse'] * weights).sum(),
        'mae': (results_df['mae'] * weights).sum(),
        'mape': (results_df['mape'] * weights).sum(),
        'smape': (results_df['smape'] * weights).sum() if 'smape' in results_df.columns else np.nan,
        'mape_log': (results_df['mape_log'] * weights).sum() if 'mape_log' in results_df.columns else np.nan,
        'mase': (results_df['mase'] * weights).sum(),
        'r2': (results_df['r2'] * weights).sum(),
    }
