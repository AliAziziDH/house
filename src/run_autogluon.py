"""
AutoGluon - Automated Ensemble for House Prices
This script uses AutoGluon's TabularPredictor to automatically train
multiple models and create an optimized ensemble.
"""

import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# ============================================
# 1. CREATE SUBMISSIONS DIRECTORY
# ============================================
os.makedirs("./submissions", exist_ok=True)

# ============================================
# 2. LOAD DATA
# ============================================
print("Loading data...")
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# ============================================
# 3. PREPARE FEATURES AND TARGET
# ============================================
# AutoGluon handles missing values and categoricals automatically.
# We only need to specify the target column.
X = train.drop(["Id", "SalePrice"], axis=1)
y = np.log1p(train["SalePrice"])  # Log-transform for RMSLE alignment
X_test = test.drop(["Id"], axis=1)

# Combine features and target into one DataFrame for AutoGluon
train_data = X.copy()
train_data["SalePrice"] = y

# ============================================
# 4. TRAIN AUTOGLUON MODEL
# ============================================
print("Training AutoGluon model (this may take 10-20 minutes)...")
predictor = TabularPredictor(
    label="SalePrice",
    problem_type="regression",
    eval_metric="root_mean_squared_error",  # RMSLE on log-transformed target
)

# num_bag_folds=5 matches project 5-fold CV; num_stack_levels=1 enables
# leakage-safe stacking via out-of-fold predictions.
# NOTE: Install ray (`pip install "ray>=2.43.0,<2.57.0"`) for parallel
# fold fitting. Without ray, folds run sequentially and need more time.
predictor.fit(
    train_data=train_data,
    presets="medium_quality",
    time_limit=1800,  # 30 min; sequential folds need ~2-3x more time than parallel
    num_bag_folds=5,
    num_stack_levels=1,
)

# ============================================
# 5. EVALUATE ON VALIDATION (Optional)
# ============================================
print("\nModel performance summary:")
print(predictor.leaderboard(silent=True))

# ============================================
# 6. PREDICT ON TEST SET
# ============================================
print("\nPredicting on test set...")
y_pred_log = predictor.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Inverse log-transform back to original scale

# ============================================
# 7. CREATE SUBMISSION
# ============================================
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": y_pred})
submission.to_csv("./submissions/submission_autogluon.csv", index=False)

print("\n✅ Submission saved to ./submissions/submission_autogluon.csv")
print(f"   Shape: {submission.shape}")
print("   First 5 rows:")
print(submission.head())
