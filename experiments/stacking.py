import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import xgboost as xgb
from catboost import CatBoostRegressor

# ============================================
# BEST PARAMETERS FROM PREVIOUS OPTIMIZATIONS
# ============================================
best_params_xgb = {
    'n_estimators': 700,
    'max_depth': 5,
    'learning_rate': 0.0186,
    'subsample': 0.6136,
    'colsample_bytree': 0.7310,
    'min_child_weight': 2,
}

best_params_catboost = {
    'iterations': 600,
    'depth': 4,
    'learning_rate': 0.094,
    'l2_leaf_reg': 3.92,
    'subsample': 0.96,
    'colsample_bylevel': 0.63,
}

# ============================================
# LOAD DATA AND MODELS
# ============================================
print("=" * 60)
print("LOADING DATA AND PREDICTIONS")
print("=" * 60)

X_train = pd.read_csv('./processed_data/X_train.csv')
X_test = pd.read_csv('./processed_data/X_test.csv')
y_train = pd.read_csv('./processed_data/y_train.csv').squeeze()

xgb_model = joblib.load('./models/xgboost_best.pkl')
catboost_model = joblib.load('./models/catboost_best.pkl')
pt = joblib.load('./models/boxcox_transformer.pkl')

print("✅ Data and models loaded successfully.")

# ============================================
# GENERATE BASE PREDICTIONS ON TRAINING SET
# ============================================
print("\n" + "=" * 60)
print("GENERATING BASE PREDICTIONS ON TRAINING SET")
print("=" * 60)

xgb_pred_train = xgb_model.predict(X_train)
catboost_pred_train = catboost_model.predict(X_train)

xgb_pred_train_original = pt.inverse_transform(xgb_pred_train.reshape(-1, 1)).flatten()
catboost_pred_train_original = pt.inverse_transform(catboost_pred_train.reshape(-1, 1)).flatten()

print("✅ Base predictions generated.")

# ============================================
# STACKING: TRAIN META-MODEL
# ============================================
print("\n" + "=" * 60)
print("TRAINING META-MODEL (LINEAR REGRESSION)")
print("=" * 60)

meta_features_train = np.column_stack([
    xgb_pred_train_original,
    catboost_pred_train_original
])

meta_model = LinearRegression()
meta_model.fit(meta_features_train, y_train)

print("✅ Meta-model trained.")
print(f"   Coefficients: {meta_model.coef_}")
print(f"   Intercept: {meta_model.intercept_:.4f}")

# ============================================
# CROSS-VALIDATION SCORE FOR STACKING
# ============================================
print("\n" + "=" * 60)
print("CROSS-VALIDATION SCORE FOR STACKING")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    xgb_model_fold = xgb.XGBRegressor(**best_params_xgb, random_state=42, verbosity=0)
    catboost_model_fold = CatBoostRegressor(**best_params_catboost, random_seed=42, verbose=False)
    
    xgb_model_fold.fit(X_train_fold, y_train_fold)
    catboost_model_fold.fit(X_train_fold, y_train_fold)
    
    xgb_pred_val = xgb_model_fold.predict(X_val_fold)
    catboost_pred_val = catboost_model_fold.predict(X_val_fold)
    
    xgb_pred_val_original = pt.inverse_transform(xgb_pred_val.reshape(-1, 1)).flatten()
    catboost_pred_val_original = pt.inverse_transform(catboost_pred_val.reshape(-1, 1)).flatten()
    
    meta_features_val = np.column_stack([xgb_pred_val_original, catboost_pred_val_original])
    
    # Replace NaN with 0 to avoid empty array error
    meta_features_val = np.nan_to_num(meta_features_val, nan=0.0)
    
    meta_model_fold = LinearRegression()
    meta_model_fold.fit(meta_features_val, y_val_fold)
    
    y_pred_meta = meta_model_fold.predict(meta_features_val)
    
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_meta))
    cv_rmse_scores.append(rmse)

print(f"   Average RMSE (Stacking): {np.mean(cv_rmse_scores):.4f} (+/- {np.std(cv_rmse_scores):.4f})")

# ============================================
# GENERATE TEST PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

xgb_pred_test = xgb_model.predict(X_test)
catboost_pred_test = catboost_model.predict(X_test)

xgb_pred_test_original = pt.inverse_transform(xgb_pred_test.reshape(-1, 1)).flatten()
catboost_pred_test_original = pt.inverse_transform(catboost_pred_test.reshape(-1, 1)).flatten()

meta_features_test = np.column_stack([
    xgb_pred_test_original,
    catboost_pred_test_original
])

stacking_pred = meta_model.predict(meta_features_test)

# ============================================
# CREATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

test_ids = pd.read_csv('./data/test.csv')['Id']
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': stacking_pred
})

submission.to_csv('submission_stacking.csv', index=False)
print("✅ Submission file saved as 'submission_stacking.csv'")
print(f"   Shape: {submission.shape}")