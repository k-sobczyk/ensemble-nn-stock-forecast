import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R-squared (R2)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }
