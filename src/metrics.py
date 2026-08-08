import numpy as np
from sklearn.metrics import mean_squared_error


def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error metric."""
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
