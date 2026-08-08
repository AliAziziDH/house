"""
XGBoost Hyperparameter Optimization with Optuna & Early Stopping
Trained on y_train_log (np.log1p(SalePrice)) directly matching Kaggle RMSLE.
"""

import os

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 15  # Optimized for fast execution in sandbox environments

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING PROCESSED DATA FOR XGBOOST")
print("=" * 60)

X_train = pd.read_csv("./processed_data/X_train.csv")
y_train_log = pd.read_csv("./processed_data/y_train_log.csv").squeeze()

# Load original raw train data to prevent target leakage in Neighborhood encoding
raw_train = pd.read_csv("./data/train.csv")
raw_train = raw_train[
    ~((raw_train["GrLivArea"] > 4000) & (raw_train["SalePrice"] < 300000))
].reset_index(drop=True)
raw_neighborhoods = raw_train["Neighborhood"]

print(f"X_train shape: {X_train.shape}")
print(f"y_train_log shape: {y_train_log.shape}")


# ============================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================
def objective(trial):
    params = {
        "objective": "reg:pseudohubererror",
        "huber_slope": trial.suggest_float("huber_slope", 0.01, 1.0, log=True),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 4
        ),  # Clamped to [3, 4] for low-variance generalization
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 12),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "n_estimators": 2000,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = (
            X_train.iloc[train_idx].copy(),
            X_train.iloc[val_idx].copy(),
        )
        y_train_fold, y_val_fold = (
            y_train_log.iloc[train_idx],
            y_train_log.iloc[val_idx],
        )

        # Leakage-Free Fold-by-Fold Neighborhood Target Rank Mapping
        fold_raw_train = raw_train.iloc[train_idx].copy()
        fold_raw_train["TotalSF"] = (
            fold_raw_train["TotalBsmtSF"]
            + fold_raw_train["1stFlrSF"]
            + fold_raw_train["2ndFlrSF"]
        )
        fold_raw_train["PricePerSF"] = (
            fold_raw_train["SalePrice"] / fold_raw_train["TotalSF"]
        )

        neigh_order = (
            fold_raw_train.groupby("Neighborhood")["PricePerSF"]
            .median()
            .sort_values()
            .index
        )
        neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

        X_train_fold["Neighborhood"] = (
            raw_neighborhoods.iloc[train_idx].map(neigh_map).fillna(13).astype(int)
        )
        X_val_fold["Neighborhood"] = (
            raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
        )

        model = XGBRegressor(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False,
        )

        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING XGBOOST OPTIMIZATION WITH EARLY STOPPING")
print("=" * 60)

os.makedirs("./experiments", exist_ok=True)
os.makedirs("./models", exist_ok=True)

study = optuna.create_study(
    direction="minimize",
    study_name="xgboost_optimization_log_target",
    storage=f"sqlite:///{os.path.abspath('./experiments/xgboost_study_log.db')}",
    load_if_exists=True,
)

study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
print(f"\n✅ Best RMSLE (log-RMSE): {study.best_value:.6f}")
print(f"✅ Best parameters: {best_params}")

# ============================================
# TRAIN FINAL MODEL ON FULL DATA
# ============================================
print("\n" + "=" * 60)
print("TRAINING FINAL XGBOOST MODEL ON FULL DATA")
print("=" * 60)

final_params = best_params.copy()
final_params.update(
    {
        "n_estimators": 2000,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }
)

tr_idx, val_idx = train_test_split(
    np.arange(len(X_train)), test_size=0.1, random_state=RANDOM_STATE
)

fold_raw_train = raw_train.iloc[tr_idx].copy()
fold_raw_train["TotalSF"] = (
    fold_raw_train["TotalBsmtSF"]
    + fold_raw_train["1stFlrSF"]
    + fold_raw_train["2ndFlrSF"]
)
fold_raw_train["PricePerSF"] = fold_raw_train["SalePrice"] / fold_raw_train["TotalSF"]

neigh_order = (
    fold_raw_train.groupby("Neighborhood")["PricePerSF"].median().sort_values().index
)
neigh_map = {n: i + 1 for i, n in enumerate(neigh_order)}

X_tr = X_train.iloc[tr_idx].copy()
X_val = X_train.iloc[val_idx].copy()
y_tr = y_train_log.iloc[tr_idx]
y_val = y_train_log.iloc[val_idx]

X_tr["Neighborhood"] = (
    raw_neighborhoods.iloc[tr_idx].map(neigh_map).fillna(13).astype(int)
)
X_val["Neighborhood"] = (
    raw_neighborhoods.iloc[val_idx].map(neigh_map).fillna(13).astype(int)
)

best_model = XGBRegressor(**final_params)
best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# Save both original naming and best rmsle versions for compatibility
joblib.dump(best_model, "./models/xgboost_best.pkl")
joblib.dump(best_model, "./models/xgboost_best_rmsle.pkl")

trials_df = study.trials_dataframe()
trials_df.to_csv("./experiments/xgboost_trials_log.csv", index=False)

# ============================================
# GENERATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

X_test = pd.read_csv("./processed_data/X_test.csv")
test_ids = pd.read_csv("./data/test.csv")["Id"]

y_pred_log = best_model.predict(X_test)
y_pred_dollars = np.expm1(y_pred_log)

os.makedirs("./submissions", exist_ok=True)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_dollars})
submission.to_csv("./submissions/submission_xgboost_log.csv", index=False)

print("✅ Submission saved to './submissions/submission_xgboost_log.csv'")
