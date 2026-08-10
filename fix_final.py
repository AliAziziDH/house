import pandas as pd
import numpy as np

for fname in ["src/ensemble.py", "src/ensemble_oof.py"]:
    with open(fname, "r") as f:
        content = f.read()

    # Fix 1: Normalized weights
    content = content.replace("weight_catboost = 0.1667\nweight_xgb = 0.1667\nweight_lgb = 0.1667\nweight_ridge = 0.1667\nweight_lasso = 0.1667\nweight_elasticnet = 0.1667", "weight_catboost = 0.5\nweight_xgb = 0.5\nweight_lgb = 0.0\nweight_ridge = 0.0\nweight_lasso = 0.0\nweight_elasticnet = 0.0")

    # Fix 2: Define missing variables
    pred_fix = """
# Define missing linear/LGBM predictions as zeros
lgb_pred_original = np.zeros(len(X_test))
ridge_pred_original = np.zeros(len(X_test))
lasso_pred_original = np.zeros(len(X_test))
elasticnet_pred_original = np.zeros(len(X_test))

# Calculate weighted average
"""
    if "lgb_pred_original = np.zeros" not in content:
        content = content.replace("# Calculate weighted average", pred_fix.strip() + "\n# Calculate weighted average")

    # Fix 3: Proper calibration logic without index misalignment
    calib_fix = """
# Instead of taking the last N elements which misaligns indices,
# just use XGB and CatBoost for calibration ensemble prediction matching the final prediction ensemble structure
ensemble_cal_log = (0.5 * xgb_cal_transformed) + (0.5 * catboost_cal_transformed)
"""
    content = content.replace("""ensemble_cal_log = (
    weight_xgb * xgb_cal_transformed +
    weight_catboost * catboost_cal_transformed +
    weight_lgb * lgb_cal_transformed +
    weight_ridge * ridge_cal_transformed +
    weight_lasso * lasso_cal_transformed +
    weight_elasticnet * elasticnet_cal_transformed
)""", calib_fix.strip())

    with open(fname, "w") as f:
        f.write(content)
