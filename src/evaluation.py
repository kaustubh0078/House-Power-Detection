import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

def evaluate_forecast(y_true, y_pred):
    """
    Compute evaluation metrics for forecasts.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary with RMSE, MAE, MAPE.
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape
    }

    return metrics
