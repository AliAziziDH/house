# .agents/skills/model_blending/scripts/run_slsqp.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.metrics import rmsle

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
        return rmsle(y_true, blend)
    
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
    
    return dict(zip(models, res.x)), res.fun