"""
LightGBM Hyperparameter Optimization with Optuna & Early Stopping
Trained on y_train_log (np.log1p(SalePrice)) directly matching Kaggle RMSLE.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import joblib
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 30

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING PROCESSED DATA FOR LIGHTGBM")
print("=" * 60)

X_train = pd.read_csv('./processed_data/X_train.csv')
y_train_log = pd.read_csv('./processed_data/y_train_log.csv').squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"y_train_log shape: {y_train_log.shape}")

# ============================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================
def objective(trial):
    params = {
        'objective': 'huber',
        'alpha': trial.suggest_float('alpha', 0.8, 0.99),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'num_leaves': trial.suggest_int('num_leaves', 15, 45),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 60),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
        'n_estimators': 2000,
        'random_state': RANDOM_STATE,
        'verbosity': -1
    }
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)

# ============================================
# RUN OPTIMIZATION
# ============================================
print("\n" + "=" * 60)
print("STARTING LIGHTGBM OPTIMIZATION WITH EARLY STOPPING")
print("=" * 60)

os.makedirs('./experiments', exist_ok=True)
os.makedirs('./models', exist_ok=True)

study = optuna.create_study(
    direction='minimize',
    study_name='lightgbm_optimization_log_target',
    storage=f'sqlite:///{os.path.abspath("./experiments/lightgbm_study_log.db")}',
    load_if_exists=True
)

study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
print(f"\n✅ Best RMSLE (log-RMSE): {study.best_value:.6f}")
print(f"✅ Best parameters: {best_params}")

# ============================================
# TRAIN FINAL MODEL ON FULL DATA
# ============================================
print("\n" + "=" * 60)
print("TRAINING FINAL LIGHTGBM MODEL ON FULL DATA")
print("=" * 60)

final_params = best_params.copy()
final_params.update({
    'n_estimators': 2000,
    'random_state': RANDOM_STATE,
    'verbosity': -1
})

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_log, test_size=0.1, random_state=RANDOM_STATE)

best_model = lgb.LGBMRegressor(**final_params)
best_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

joblib.dump(best_model, './models/lightgbm_best.pkl')
trials_df = study.trials_dataframe()
trials_df.to_csv('./experiments/lightgbm_trials_log.csv', index=False)

# ============================================
# GENERATE SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

X_test = pd.read_csv('./processed_data/X_test.csv')
test_ids = pd.read_csv('./data/test.csv')['Id']

y_pred_log = best_model.predict(X_test)
y_pred_dollars = np.expm1(y_pred_log)

os.makedirs('./submissions', exist_ok=True)
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred_dollars})
submission.to_csv('./submissions/submission_lightgbm_log.csv', index=False)

print("✅ Submission saved to './submissions/submission_lightgbm_log.csv'")