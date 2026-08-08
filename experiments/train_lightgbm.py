"""
LightGBM Training and Optimization with Optuna
"""

import os

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer

from src.metrics import rmsle

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING PREPROCESSED DATA")
print("=" * 60)

X_train = pd.read_csv("./processed_data/X_train.csv")
y_train = pd.read_csv("./processed_data/y_train.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ============================================
# BOX-COX TRANSFORMATION
# ============================================
print("\n" + "=" * 60)
print("APPLYING BOX-COX TRANSFORMATION")
print("=" * 60)

pt = PowerTransformer(method="box-cox")
y_transformed = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
print(f"Skewness after Box-Cox: {pd.Series(y_transformed).skew():.4f}")


# ============================================
# OPTUNA OBJECTIVE FOR LIGHTGBM
# ============================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "random_state": 42,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed[train_idx], y_transformed[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        # Inverse transform
        y_pred_original = pt.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_val_original = pt.inverse_transform(y_val_fold.reshape(-1, 1)).flatten()

        score = rmsle(y_val_original, y_pred_original)
        rmse_scores.append(score)

    return np.mean(rmse_scores)


# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING LIGHTGBM OPTIMIZATION")
print("=" * 60)

os.makedirs("./experiments", exist_ok=True)

study = optuna.create_study(
    direction="minimize",
    study_name="lightgbm_optimization",
    storage=f"sqlite:///{os.path.abspath('./experiments/lightgbm_study.db')}",
    load_if_exists=True,
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

best_params = study.best_params
best_model = lgb.LGBMRegressor(**best_params, random_state=42, verbosity=-1)
best_model.fit(X_train, y_transformed)

# Save model
os.makedirs("./models", exist_ok=True)
joblib.dump(best_model, "./models/lightgbm_best.pkl")

# Save trials
trials_df = study.trials_dataframe()
trials_df.to_csv("./experiments/lightgbm_trials.csv", index=False)

print(f"✅ Best RMSE: {study.best_value:.4f}")
print(f"✅ Best parameters: {best_params}")
print("✅ Model saved to './models/lightgbm_best.pkl'")
print("✅ Trials saved to './experiments/lightgbm_trials.csv'")

# ============================================
# GENERATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

X_test = pd.read_csv("./processed_data/X_test.csv")
y_pred_transformed = best_model.predict(X_test)
y_pred_original = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()

submission = pd.DataFrame(
    {"Id": pd.read_csv("./data/test.csv")["Id"], "SalePrice": y_pred_original}
)
submission.to_csv("submission_lightgbm.csv", index=False)

print("✅ Submission file saved as 'submission_lightgbm.csv'")
print(f"   Shape: {submission.shape}")
