"""
CatBoost Training and Optimization with Optuna
This script uses raw data (before one-hot encoding) to leverage CatBoost's native categorical feature support.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import joblib
import os

# ============================================
# LOAD RAW DATA (BEFORE ONE-HOT ENCODING)
# ============================================
print("=" * 60)
print("LOADING RAW DATA (BEFORE ONE-HOT ENCODING)")
print("=" * 60)

# Load the raw preprocessed data from preprocess.py (before one-hot encoding)
# For this, we need to run preprocess.py and save the data before one-hot encoding.
# We'll use the data saved in processed_data/ but before one-hot encoding.

# To avoid re-running everything, we'll load the original train and test data
# and apply only the preprocessing steps (missing value imputation and feature engineering).
# For simplicity, we'll use the data from preprocess.py but before one-hot encoding.

# If you haven't saved the data before one-hot encoding, you can load the original data
# and apply the preprocessing steps manually. We'll use the processed data from preprocess.py
# but we'll skip the one-hot encoding step.

# For this script, we assume you have the data before one-hot encoding saved as:
# - X_train_raw.csv
# - X_test_raw.csv
# If not, we'll load the original data and apply preprocessing.

print("Loading raw preprocessed data (before one-hot encoding)...")
X_train_raw = pd.read_csv('./processed_data/X_train_raw.csv')
X_test_raw = pd.read_csv('./processed_data/X_test_raw.csv')
y_train = pd.read_csv('./processed_data/y_train.csv').squeeze()

print(f"X_train_raw shape: {X_train_raw.shape}")
print(f"X_test_raw shape: {X_test_raw.shape}")
print(f"y_train shape: {y_train.shape}")

# Identify categorical features (object type)
cat_features = X_train_raw.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features: {len(cat_features)}")
print(f"First 5: {cat_features[:5]}")

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
# OPTUNA OBJECTIVE FOR CATBOOST
# ============================================
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'random_seed': 42,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train_raw):
        X_train_fold, X_val_fold = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]
        
        # Fit with categorical features
        model.fit(
            X_train_fold, y_train_fold,
            cat_features=cat_features,
            eval_set=(X_val_fold, y_val_fold),
            verbose=False
        )
        
        y_pred = model.predict(X_val_fold)
        
        # Inverse transform
        y_pred_original = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING CATBOOST OPTIMIZATION")
print("=" * 60)

os.makedirs('./experiments', exist_ok=True)

study = optuna.create_study(
    direction='minimize',
    study_name='catboost_optimization',
    storage=f'sqlite:///{os.path.abspath("./experiments/catboost_study.db")}',
    load_if_exists=True
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

best_params = study.best_params
best_model = CatBoostRegressor(**best_params, random_seed=42, verbose=False)
best_model.fit(X_train_raw, y_transformed, cat_features=cat_features)

# Save model
os.makedirs('./models', exist_ok=True)
joblib.dump(best_model, './models/catboost_best.pkl')

# Save trials
trials_df = study.trials_dataframe()
trials_df.to_csv('./experiments/catboost_trials.csv', index=False)

print(f"✅ Best RMSE: {study.best_value:.4f}")
print(f"✅ Best parameters: {best_params}")
print("✅ Model saved to './models/catboost_best.pkl'")
print("✅ Trials saved to './experiments/catboost_trials.csv'")

# ============================================
# GENERATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

y_pred_transformed = best_model.predict(X_test_raw)
y_pred_original = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

submission = pd.DataFrame({
    'Id': pd.read_csv('./data/test.csv')['Id'],
    'SalePrice': y_pred_original
})
submission.to_csv('submission_catboost.csv', index=False)

print("✅ Submission file saved as 'submission_catboost.csv'")
print(f"   Shape: {submission.shape}")