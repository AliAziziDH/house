import joblib
import pandas as pd

# ============================================
# LOAD MODELS AND DATA
# ============================================
print("=" * 60)
print("LOADING MODELS AND DATA")
print("=" * 60)

# Load test data
import numpy as np

X_test = pd.read_csv("./processed_data/X_test.csv")
X_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]

raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train[
    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))
].reset_index(drop=True)
raw_test_neighborhoods = pd.read_csv("./data/test.csv")["Neighborhood"]

raw_train["TotalSF"] = (
    raw_train["TotalBsmtSF"] + raw_train["1stFlrSF"] + raw_train["2ndFlrSF"]
)
raw_train["PricePerSF"] = raw_train["SalePrice"] / raw_train["TotalSF"]
neigh_order = (
    raw_train.groupby("Neighborhood")["PricePerSF"].median().sort_values().index
)
neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

X_test["Neighborhood"] = raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)
X_test_raw["Neighborhood"] = raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)

# Load trained models (we only use the best ones)
xgb_model = joblib.load("./models/xgboost_best_rmsle.pkl")
catboost_model = joblib.load("./models/catboost_best_rmsle.pkl")

# Load the transformer (Box-Cox)
print("✅ Models loaded successfully.")

# ============================================
# GENERATE PREDICTIONS
# ============================================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

# Predict from each model (in transformed scale)
xgb_pred_transformed = xgb_model.predict(X_test)
catboost_pred_transformed = catboost_model.predict(X_test_raw)

# Inverse transform to original scale (dollars)
xgb_pred_original = np.expm1(xgb_pred_transformed)
catboost_pred_original = np.expm1(catboost_pred_transformed)

print("✅ Predictions generated for both models.")

# ============================================
# WEIGHTED AVERAGE ENSEMBLE
# ============================================
print("\n" + "=" * 60)
print("ENSEMBLE: WEIGHTED AVERAGE")
print("=" * 60)

# Best weights from optimization (aligned with README)
weight_xgb = 0.5003
weight_catboost = 0.4997

# Calculate weighted average
ensemble_pred = (
    weight_xgb * xgb_pred_original + weight_catboost * catboost_pred_original
)

ensemble_pred = np.clip(ensemble_pred, 34900, 755000)

print(f"Weights: XGBoost = {weight_xgb:.2f}, CatBoost = {weight_catboost:.2f}")

# ============================================
# INDUCTIVE CONFORMAL PREDICTION (ICP)
# ============================================
print("\n" + "=" * 60)
print("CALCULATING CONFORMAL PREDICTION INTERVALS")
print("=" * 60)

from sklearn.model_selection import train_test_split

X_train = pd.read_csv("./processed_data/X_train.csv")
X_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")
y_train_log = pd.read_csv("./processed_data/y_train_log.csv").squeeze()

X_train["Neighborhood"] = raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)
X_train_raw["Neighborhood"] = raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)

# Recreate 10% calibration set split
_, X_cal, _, y_cal_log = train_test_split(
    X_train, y_train_log, test_size=0.1, random_state=42
)
_, X_cal_raw, _, _ = train_test_split(
    X_train_raw, y_train_log, test_size=0.1, random_state=42
)

# Generate ensemble predictions on calibration set
xgb_cal_transformed = xgb_model.predict(X_cal)
catboost_cal_transformed = catboost_model.predict(X_cal_raw)

# Calculate calibration ensemble prediction in log space
ensemble_cal_log = (weight_xgb * xgb_cal_transformed) + (weight_catboost * catboost_cal_transformed)

# Calculate absolute residuals in log space
R = np.abs(y_cal_log.values - ensemble_cal_log)

# Calculate non-conformity quantile (alpha = 0.05)
alpha = 0.05
n_cal = len(R)
quantile_val = (1.0 - alpha) * (1.0 + 1.0 / n_cal)
q = np.quantile(R, min(quantile_val, 1.0))

# Convert test predictions back to log space to calculate bounds
ensemble_pred_log = np.log1p(ensemble_pred)

# Calculate bounds and convert back to dollars
lower_bounds = np.expm1(ensemble_pred_log - q)
upper_bounds = np.expm1(ensemble_pred_log + q)

print(f"✅ Calibration Quantile (q) = {q:.5f}")

# ============================================
# CREATE SUBMISSION FILES
# ============================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILES")
print("=" * 60)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": ensemble_pred})
submission_intervals = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": ensemble_pred,
    "Price_Lower_Bound": lower_bounds,
    "Price_Upper_Bound": upper_bounds
})

import os

os.makedirs("./submissions", exist_ok=True)

submission.to_csv("./submissions/submission_ensemble_oof.csv", index=False)
submission_intervals.to_csv("./submissions/submission_oof_intervals.csv", index=False)

print("✅ Standard submission saved to './submissions/submission_ensemble_oof.csv'")
print("✅ ICP intervals saved to './submissions/submission_oof_intervals.csv'")
