"""
AutoGluon training script for Colab execution.
Data is expected at: train.csv, test.csv (VM working directory)

Quality Gates:
  - Target transformed with np.log1p pre-training (RMSLE alignment).
  - Predictions inverse-transformed with np.expm1 post-prediction.
  - eval_metric strictly set to root_mean_squared_error.
  - 2-fold bagging, no stacking (fast mode — ~3 min budget).
"""
# ── Step 1: Install dependencies only when not already present ────────────────
# Skipped entirely on a reused session — saves 2-3 min.
import subprocess
import sys

try:
    import autogluon.tabular  # noqa: F401 — fast check, no side effects
    print("AutoGluon already installed — skipping install step. ⚡\n")
except ImportError:
    print("Installing uv + AutoGluon + CatBoost on Colab VM (fast mode)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    subprocess.run(
        ["uv", "pip", "install", "-q", "--system",
         "autogluon.tabular[catboost,lightgbm,xgboost]"],
        check=True,
    )
    print("Installation complete.\n")

# ── Step 2: Standard imports (autogluon now guaranteed to be present) ─────────
import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor  # safe to import after pip install

os.makedirs("submissions", exist_ok=True)

# ── Step 3: Load data ─────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

X      = train.drop(["Id", "SalePrice"], axis=1)
y      = np.log1p(train["SalePrice"])   # log-transform for RMSLE alignment
X_test = test.drop(["Id"], axis=1)

train_data = X.copy()
train_data["SalePrice"] = y

# ── Step 4: Train ─────────────────────────────────────────────────────────────
print("Training AutoGluon (fast mode: 2-fold, 3-min budget, no stacking)...")
predictor = TabularPredictor(
    label="SalePrice",
    problem_type="regression",
    eval_metric="root_mean_squared_error",  # RMSE on log-target ≡ RMSLE
)
predictor.fit(
    train_data=train_data,
    presets="medium_quality",
    time_limit=180,           # 3 min — fast turnaround for iteration
    num_bag_folds=2,          # reduced from 5 to cut CV overhead
    num_stack_levels=0,       # no stacking — biggest single speed lever
)

# ── Step 5: Evaluate & report ─────────────────────────────────────────────────
print("\n--- Leaderboard ---")
lb = predictor.leaderboard(silent=True)
print(lb[["model", "score_val", "fit_time"]].to_string())

# ── Step 6: Predict & save submission ─────────────────────────────────────────
print("\nPredicting on test set...")
y_pred_log = predictor.predict(X_test)
y_pred     = np.expm1(y_pred_log)        # inverse log-transform

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": y_pred})
submission.to_csv("submissions/submission_autogluon.csv", index=False)
print(f"Saved: submissions/submission_autogluon.csv  shape={submission.shape}")
print(f"SalePrice — min: {y_pred.min():.0f}  median: {y_pred.median():.0f}  max: {y_pred.max():.0f}")
