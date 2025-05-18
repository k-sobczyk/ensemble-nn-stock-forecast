import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    y_true_filtered = y_true[non_zero_mask]
    y_pred_filtered = y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Calculates Mean Absolute Scaled Error"""
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


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_raw: np.ndarray):
    """Calculates and prints RMSE, MAE, R2, MAPE, MASE"""
    if y_true.size == 0 or y_pred.size == 0:
        print('\nEvaluation Metrics: Cannot calculate metrics - No prediction data available.')
        return

    if y_true.size != y_pred.size:
        print(
            f'\nError: Mismatch in size of y_true ({y_true.size}) and y_pred ({y_pred.size}). Cannot calculate metrics.'
        )
        return

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r2 = max(-float('inf'), min(1.0, r2))  # Clamp R2 score

    mape = calculate_mape(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred, y_train_raw)

    print('\n--- Evaluation Metrics ---')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE:  {mae:.4f}')
    print(f'R²:   {r2:.4f}')
    print(f'MAPE: {mape:.2f}%' if not np.isnan(mape) else 'MAPE: N/A (Check for zeros in actuals)')
    print(f'MASE: {mase:.4f}' if not np.isnan(mase) and not np.isinf(mase) else f'MASE: N/A ({mase})')
    print('--------------------------')
