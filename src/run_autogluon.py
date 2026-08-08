"""
AutoGluon - Automated Ensemble for House Prices
This script uses AutoGluon's TabularPredictor to automatically train
multiple models and create an optimized ensemble.
"""

import os

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
y = train["SalePrice"]
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
    eval_metric="smape",  # Using smape as closest to RMSLE
)

# Use 'medium_quality' preset for a balance of speed and accuracy.
# If you want faster results, change to 'low_quality'.
predictor.fit(
    train_data=train_data,  # Now includes the target column
    presets="medium_quality",
    time_limit=600,  # 10 minutes; increase if needed
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
y_pred = predictor.predict(X_test)

# ============================================
# 7. CREATE SUBMISSION
# ============================================
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": y_pred})
submission.to_csv("./submissions/submission_autogluon.csv", index=False)

print("\n✅ Submission saved to ./submissions/submission_autogluon.csv")
print(f"   Shape: {submission.shape}")
print("   First 5 rows:")
print(submission.head())
