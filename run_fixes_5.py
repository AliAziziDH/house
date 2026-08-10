import pandas as pd
import numpy as np

for fname, std_name, int_name in [("src/ensemble.py", "submission_ensemble_final.csv", "submission_with_intervals.csv"),
                                  ("src/ensemble_oof.py", "submission_ensemble_oof.csv", "submission_oof_intervals.csv")]:
    with open(fname, "r") as f:
        content = f.read()

    # Load raw
    content = content.replace('X_test = pd.read_csv("./processed_data/X_test.csv")',
                              'import numpy as np\nX_test = pd.read_csv("./processed_data/X_test.csv")\nX_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")')

    content = content.replace('test_ids = pd.read_csv("./data/test.csv")["Id"]',
                              'test_ids = pd.read_csv("./data/test.csv")["Id"]\n\nraw_train = pd.read_csv("./data/train.csv")\nraw_train = raw_train[\n    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 200000))\n].reset_index(drop=True)\nraw_test_neighborhoods = pd.read_csv("./data/test.csv")["Neighborhood"]\n\nraw_train["TotalSF"] = (\n    raw_train["TotalBsmtSF"] + raw_train["1stFlrSF"] + raw_train["2ndFlrSF"]\n)\nraw_train["PricePerSF"] = raw_train["SalePrice"] / raw_train["TotalSF"]\nneigh_order = (\n    raw_train.groupby("Neighborhood")["PricePerSF"].median().sort_values().index\n)\nneigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}\n\nX_test["Neighborhood"] = raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)\nX_test_raw["Neighborhood"] = raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)')

    content = content.replace('catboost_pred_transformed = catboost_model.predict(X_test)',
                              'catboost_pred_transformed = catboost_model.predict(X_test_raw)')

    # ICP fixes
    icp_code = """
# ============================================
# INDUCTIVE CONFORMAL PREDICTION (ICP)
# ============================================
print("\\n" + "=" * 60)
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
print("\\n" + "=" * 60)
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
"""

    final_code = icp_code + f"""
submission.to_csv("./submissions/{std_name}", index=False)
submission_intervals.to_csv("./submissions/{int_name}", index=False)

print("✅ Standard submission saved to './submissions/{std_name}'")
print("✅ ICP intervals saved to './submissions/{int_name}'")
"""
    to_replace = """
# ============================================
# CREATE SUBMISSION
# ============================================
print("\\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": ensemble_pred})

import os

os.makedirs("./submissions", exist_ok=True)
"""
    content = content.split(to_replace)[0] + final_code

    # inverse transform logic
    if "pt.inverse_transform" in content:
        content = content.replace('pt.inverse_transform(xgb_pred_transformed.reshape(-1, 1)).flatten()', 'np.expm1(xgb_pred_transformed)')
        content = content.replace('pt.inverse_transform(\n    catboost_pred_transformed.reshape(-1, 1)\n).flatten()', 'np.expm1(catboost_pred_transformed)')
        content = content.replace('pt = joblib.load("./models/boxcox_transformer.pkl")\n\nprint("✅ Models and transformer loaded successfully.")', 'print("✅ Models loaded successfully.")')

    # weights logic
    content = content.replace('weight_xgb = 0.64', 'weight_xgb = 0.5003')
    content = content.replace('weight_catboost = 0.36', 'weight_catboost = 0.4997')

    # bounds
    content = content.replace('print(f"Weights: XGBoost = {weight_xgb:.2f}, CatBoost = {weight_catboost:.2f}")', 'ensemble_pred = np.clip(ensemble_pred, 34900, 755000)\n\nprint(f"Weights: XGBoost = {weight_xgb:.2f}, CatBoost = {weight_catboost:.2f}")')

    with open(fname, "w") as f:
        f.write(content)
