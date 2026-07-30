"""
CatBoost Optimization with Optuna
Uses RMSLE (Root Mean Squared Log Error) as the evaluation metric,
which aligns with the competition's official metric.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 50

# ============================================
# RMSLE METRIC
# ============================================
def rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error
    This is the official metric for the House Prices competition.
    """
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

X_train = pd.read_csv('./processed_data/X_train.csv')
y_train = pd.read_csv('./processed_data/y_train.csv').squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ============================================
# BOX-COX TRANSFORMATION (still used for model input)
# ============================================
print("\n" + "=" * 60)
print("APPLYING BOX-COX TRANSFORMATION")
print("=" * 60)

pt = PowerTransformer(method='box-cox')
y_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
print(f"Skewness after Box-Cox: {pd.Series(y_transformed).skew():.4f}")

# ============================================
# OPTUNA OBJECTIVE FUNCTION (with RMSLE)
# ============================================
def objective(trial):
    """
    Objective function for Optuna to minimize RMSLE on cross-validation.
    """
    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'random_seed': RANDOM_STATE,
        'verbose': False
    }
    
    # Create model
    model = CatBoostRegressor(**params)
    
    # Cross-validation
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmsle_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]
        
        # Train model on transformed target
        model.fit(X_train_fold, y_train_fold, verbose=False)
        
        # Predict on validation fold
        y_pred_transformed = model.predict(X_val_fold)
        
        # Inverse transform to original scale
        y_pred_original = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()
        
        # Calculate RMSLE on original scale
        rmsle_score = rmsle(y_val_original, y_pred_original)
        rmsle_scores.append(rmsle_score)
    
    avg_rmsle = np.mean(rmsle_scores)
    return avg_rmsle

# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING CATBOOST OPTIMIZATION (RMSLE)")
print("=" * 60)

# Create directories
os.makedirs('./experiments', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# Create study
study = optuna.create_study(
    direction='minimize',
    study_name='catboost_optimization_rmsle',
    storage=f'sqlite:///{os.path.abspath("./experiments/catboost_study_rmsle.db")}',
    load_if_exists=True
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

best_params = study.best_params
best_model = CatBoostRegressor(**best_params, random_seed=RANDOM_STATE, verbose=False)
best_model.fit(X_train, y_transformed)

# Save model and transformer
joblib.dump(best_model, './models/catboost_best_rmsle.pkl')
joblib.dump(pt, './models/boxcox_transformer.pkl')

# Save all trial results
trials_df = study.trials_dataframe()
trials_df.to_csv('./experiments/catboost_trials_rmsle.csv', index=False)

print(f"✅ Best RMSLE: {study.best_value:.6f}")
print(f"✅ Best parameters: {best_params}")
print("✅ Model saved to './models/catboost_best_rmsle.pkl'")
print("✅ Trials saved to './experiments/catboost_trials_rmsle.csv'")

# ============================================
# GENERATE SUBMISSION WITH BEST MODEL
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

X_test = pd.read_csv('./processed_data/X_test.csv')
test_ids = pd.read_csv('./data/test.csv')['Id']

y_pred_transformed = best_model.predict(X_test)
y_pred_original = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred_original
})
submission.to_csv('./submissions/submission_catboost_rmsle.csv', index=False)

print("✅ Submission saved to './submissions/submission_catboost_rmsle.csv'")
print(f"   Shape: {submission.shape}")
print("   First 5 rows:")
print(submission.head())

print("\n" + "=" * 60)
print("CATBOOST OPTIMIZATION COMPLETED")
print("=" * 60)