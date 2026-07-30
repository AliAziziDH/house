"""
Stacking with OOF (Out-of-Fold) Predictions
This is a corrected implementation that prevents data leakage and uses proper cross-validation.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5

# Best parameters from previous optimizations
best_params_xgb = {
    'n_estimators': 700,
    'max_depth': 5,
    'learning_rate': 0.0186,
    'subsample': 0.6136,
    'colsample_bytree': 0.7310,
    'min_child_weight': 2,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

best_params_catboost = {
    'iterations': 600,
    'depth': 4,
    'learning_rate': 0.094,
    'l2_leaf_reg': 3.92,
    'subsample': 0.96,
    'colsample_bylevel': 0.63,
    'random_seed': RANDOM_STATE,
    'verbose': False
}

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

X_train = pd.read_csv('./processed_data/X_train.csv')
y_train = pd.read_csv('./processed_data/y_train.csv').squeeze()
X_test = pd.read_csv('./processed_data/X_test.csv')
test_ids = pd.read_csv('./data/test.csv')['Id']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")

# Load Box-Cox transformer for inverse transform
pt = joblib.load('./models/boxcox_transformer.pkl')

# ============================================
# RMSLE METRIC
# ============================================
def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error"""
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# ============================================
# OOF PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING OOF PREDICTIONS")
print("=" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Arrays to store OOF predictions
xgb_oof = np.zeros(len(X_train))
cat_oof = np.zeros(len(X_train))

# Lists to store models for test predictions
xgb_models = []
catboost_models = []

print(f"Training {N_FOLDS} folds for each model...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")
    
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(**best_params_xgb)
    xgb_model.fit(X_train_fold, y_train_fold)
    xgb_oof[val_idx] = xgb_model.predict(X_val_fold)
    xgb_models.append(xgb_model)
    
    # Train CatBoost
    cat_model = CatBoostRegressor(**best_params_catboost)
    cat_model.fit(X_train_fold, y_train_fold)
    cat_oof[val_idx] = cat_model.predict(X_val_fold)
    catboost_models.append(cat_model)
    
    print(f"    XGBoost OOF RMSLE: {rmsle(y_val_fold, xgb_oof[val_idx]):.6f}")
    print(f"    CatBoost OOF RMSLE: {rmsle(y_val_fold, cat_oof[val_idx]):.6f}")

# ============================================
# TRAIN META-MODEL ON OOF PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("TRAINING META-MODEL")
print("=" * 60)

# Stack OOF predictions as meta-features
meta_features = np.column_stack([xgb_oof, cat_oof])

# Train meta-model (Linear Regression)
meta_model = LinearRegression()
meta_model.fit(meta_features, y_train)

print(f"Meta-model coefficients: {meta_model.coef_}")
print(f"Meta-model intercept: {meta_model.intercept_:.4f}")

# Evaluate meta-model on OOF predictions
oof_pred = meta_model.predict(meta_features)
oof_rmsle = rmsle(y_train, oof_pred)
print(f"\nOOF RMSLE (meta-model on OOF): {oof_rmsle:.6f}")

# ============================================
# TRAIN FINAL MODELS ON FULL DATA
# ============================================
print("\n" + "=" * 60)
print("TRAINING FINAL MODELS ON FULL DATA")
print("=" * 60)

xgb_final = xgb.XGBRegressor(**best_params_xgb)
xgb_final.fit(X_train, y_train)
print("✅ XGBoost trained on full data.")

cat_final = CatBoostRegressor(**best_params_catboost)
cat_final.fit(X_train, y_train)
print("✅ CatBoost trained on full data.")

# ============================================
# GENERATE TEST PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

# Predict with base models on test set
xgb_test_pred = xgb_final.predict(X_test)
cat_test_pred = cat_final.predict(X_test)

# Stack test predictions
meta_features_test = np.column_stack([xgb_test_pred, cat_test_pred])

# Final stacking prediction
stacking_pred = meta_model.predict(meta_features_test)

# ============================================
# CREATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': stacking_pred
})

submission.to_csv('./submissions/submission_stacking_final.csv', index=False)

print("✅ Submission saved to ./submissions/submission_stacking_final.csv")
print(f"   Shape: {submission.shape}")
print("\n   First 5 rows:")
print(submission.head())

# ============================================
# COMPARE WITH WEIGHTED ENSEMBLE
# ============================================
print("\n" + "=" * 60)
print("COMPARISON WITH WEIGHTED ENSEMBLE")
print("=" * 60)

# Weighted ensemble predictions (using best weights)
weight_xgb = 0.64
weight_catboost = 0.36
ensemble_pred = weight_xgb * xgb_oof + weight_catboost * cat_oof
ensemble_rmsle = rmsle(y_train, ensemble_pred)

print(f"Weighted Ensemble OOF RMSLE: {ensemble_rmsle:.6f}")
print(f"Stacking OOF RMSLE: {oof_rmsle:.6f}")

if oof_rmsle < ensemble_rmsle:
    print("✅ Stacking performs better on OOF!")
else:
    print("⚠️ Weighted ensemble still performs better on OOF.")

print("\n" + "=" * 60)
print("STACKING COMPLETED")
print("=" * 60)