from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def calculate_rmse(y_true, y_pred):
    """Count Root Mean Squared Error (RMSE)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def calculate_mae(y_true, y_pred):
    """Count Mean Absolute Error (MAE)"""
    mae = mean_absolute_error(y_true, y_pred)
    return mae


def calculate_r2(y_true, y_pred):
    """Count Coefficient of Determination (R²)"""
    r2 = r2_score(y_true, y_pred)
    return r2
