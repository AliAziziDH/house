import pandas as pd
import numpy as np

for fname, std_name, int_name in [("src/ensemble.py", "submission_ensemble_final.csv", "submission_with_intervals.csv"),
                                  ("src/ensemble_oof.py", "submission_ensemble_oof.csv", "submission_oof_intervals.csv")]:
    with open(fname, "r") as f:
        content = f.read()

    # Load raw
    content = content.replace('X_test = pd.read_csv("./processed_data/X_test.csv")',
                              'X_test = pd.read_csv("./processed_data/X_test.csv")\\nX_test_raw = pd.read_csv("./processed_data/X_test_raw.csv")')

    content = content.replace('catboost_pred_transformed = catboost_model.predict(X_test)',
                              'X_test_raw["Neighborhood"] = raw_test_neighborhoods.map(neigh_map).fillna(13).astype(int)\\ncatboost_pred_transformed = catboost_model.predict(X_test_raw)')

    # ICP fixes
    content = content.replace('X_train = pd.read_csv("./processed_data/X_train.csv")',
                              'X_train = pd.read_csv("./processed_data/X_train.csv")\\nX_train_raw = pd.read_csv("./processed_data/X_train_raw.csv")')
    content = content.replace('_, X_cal, _, y_cal_log = train_test_split(',
                              '_, X_cal_raw, _, _ = train_test_split(X_train_raw, y_train_log, test_size=0.1, random_state=42)\\n_, X_cal, _, y_cal_log = train_test_split(')
    content = content.replace('X_train["Neighborhood"] = raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)',
                              'X_train["Neighborhood"] = raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)\\nX_train_raw["Neighborhood"] = raw_train["Neighborhood"].map(neigh_map).fillna(13).astype(int)')
    content = content.replace('catboost_cal_transformed = catboost_model.predict(X_cal)',
                              'catboost_cal_transformed = catboost_model.predict(X_cal_raw)')

    # inverse transform logic
    if "pt.inverse_transform" in content:
        content = content.replace('pt.inverse_transform(xgb_pred_transformed.reshape(-1, 1)).flatten()', 'np.expm1(xgb_pred_transformed)')
        content = content.replace('pt.inverse_transform(\\n    catboost_pred_transformed.reshape(-1, 1)\\n).flatten()', 'np.expm1(catboost_pred_transformed)')
        content = content.replace('pt = joblib.load("./models/boxcox_transformer.pkl")\\n\\nprint("✅ Models and transformer loaded successfully.")', 'print("✅ Models loaded successfully.")')

    # weights logic
    content = content.replace('weight_xgb = 0.64', 'weight_xgb = 0.5003')
    content = content.replace('weight_catboost = 0.36', 'weight_catboost = 0.4997')
    content = content.replace('weight_xgb = 0.1667', 'weight_xgb = 0.5003')
    content = content.replace('weight_catboost = 0.1665', 'weight_catboost = 0.4997')
    content = content.replace('weight_lgb = 0.1667\\nweight_ridge = 0.1667\\nweight_lasso = 0.1667\\nweight_elasticnet = 0.1667\\n', '')

    with open(fname, "w") as f:
        f.write(content)
