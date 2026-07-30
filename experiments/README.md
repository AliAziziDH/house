# Experiments Directory

This folder contains experiment scripts, study databases, and trial CSVs produced during model development.

Purpose:
- Keep reproducible experiment code here (scripts used to run Optuna, stacking, and other experiments).
- Archive large experiment artifacts (database files and trial CSVs) to avoid cluttering the main experiments folder.

Canonical scripts (keep and maintain):
- `optimize_xgboost.py` — XGBoost Optuna optimization (RMSLE).
- `train_catboost.py` — CatBoost Optuna optimization using raw data and `cat_features`.
- `train_lightgbm.py` — LightGBM Optuna optimization.
- `find_ensemble_weights.py` / `optimize_ensemble_weights.py` — OOF-weight and submission generation utilities.
- `stacking.py` — OOF stacking experiment and meta-model.
- `encode.py`, `test_transformations.py` — utility scripts for encoding and testing target transforms.

Artifacts and archives:
- Large files such as `*.db` and `*_trials.csv` have been moved to `experiments/archive/` to keep this directory tidy.

If you need to re-run or inspect archived experiments, restore files from `experiments/archive/` or re-run the corresponding optimization script to re-generate study files.
