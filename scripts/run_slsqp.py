# .agents/skills/model_blending/scripts/run_slsqp.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_weights(oof_predictions: pd.DataFrame, y_true: np.ndarray):
    """
    Optimizes blending weights using SLSQP solver to minimize RMSLE.
    """
    models = oof_predictions.columns
    n_models = len(models)
    
    # Target values are already in normal scale (untransformed)
    # Objective function
    def objective(weights):
        blend = np.zeros_like(y_true, dtype=float)
        for i, col in enumerate(models):
            blend += weights[i] * oof_predictions[col].values

        log_y_true = np.log1p(y_true)
        log_ensemble_pred = np.log1p(blend)
        return np.sum((log_y_true - log_ensemble_pred) ** 2)
    
    # Constraints & Bounds
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    initial_weights = np.ones(n_models) / n_models
    
    res = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    
    return dict(zip(models, res.x)), np.sqrt(res.fun / len(y_true))