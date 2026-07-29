"""
XGBoost Hyperparameter Optimization with Optuna
This script performs systematic hyperparameter tuning using Optuna
and logs all trials for later analysis.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

X_train = pd.read_csv('./processed_data/X_train.csv')
y_train = pd.read_csv('./processed_data/y_train.csv').squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ============================================
# BOX-COX TRANSFORMATION
# ============================================
print("\n" + "=" * 60)
print("APPLYING BOX-COX TRANSFORMATION")
print("=" * 60)

pt = PowerTransformer(method='box-cox')
y_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
print(f"Skewness after Box-Cox: {pd.Series(y_transformed).skew():.4f}")

# ============================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================
def objective(trial):
    """
    Objective function for Optuna to minimize RMSE on cross-validation.
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'verbosity': 0
    }
    
    # Create model
    model = XGBRegressor(**params)
    
    # Cross-validation (5-fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        
        # Inverse transform to original scale
        y_pred_original = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
        rmse_scores.append(rmse)
    
    avg_rmse = np.mean(rmse_scores)
    return avg_rmse

# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING OPTUNA OPTIMIZATION")
print("=" * 60)

# Create study and optimize
study = optuna.create_study(
    direction='minimize',
    study_name='xgboost_optimization',
    storage='sqlite:///experiments/xgboost_study.db',
    load_if_exists=True
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Create directories if they don't exist
os.makedirs('./experiments', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# Save best parameters and model
best_params = study.best_params
best_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
best_model.fit(X_train, y_transformed)

# Save model and transformer
joblib.dump(best_model, './models/xgboost_best.pkl')
joblib.dump(pt, './models/boxcox_transformer.pkl')

# Save all trial results
trials_df = study.trials_dataframe()
trials_df.to_csv('./experiments/xgboost_trials.csv', index=False)

print(f"✅ Best RMSE: {study.best_value:.4f}")
print(f"✅ Best parameters: {best_params}")
print("✅ Model saved to './models/xgboost_best.pkl'")
print("✅ Trials saved to './experiments/xgboost_trials.csv'")

# ============================================
# GENERATE SUBMISSION WITH BEST MODEL
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION WITH BEST MODEL")
print("=" * 60)

X_test = pd.read_csv('./processed_data/X_test.csv')
y_pred_transformed = best_model.predict(X_test)
y_pred_original = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

submission = pd.DataFrame({
    'Id': pd.read_csv('./data/test.csv')['Id'],
    'SalePrice': y_pred_original
})
submission.to_csv('submission_optimized.csv', index=False)

print("✅ Submission file saved as 'submission_optimized.csv'")
print(f"   Shape: {submission.shape}")