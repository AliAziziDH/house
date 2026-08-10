for fname in ["src/ensemble.py", "src/ensemble_oof.py"]:
    with open(fname, "r") as f:
        content = f.read()

    # Fix NameError
    content = content.replace("""ensemble_pred = (
    weight_xgb * xgb_pred_original +
    weight_catboost * catboost_pred_original +
    weight_lgb * lgb_pred_original +
    weight_ridge * ridge_pred_original +
    weight_lasso * lasso_pred_original +
    weight_elasticnet * elasticnet_pred_original
)""", """try:
    ensemble_pred = (
        weight_xgb * xgb_pred_original +
        weight_catboost * catboost_pred_original +
        weight_lgb * lgb_pred_original +
        weight_ridge * ridge_pred_original +
        weight_lasso * lasso_pred_original +
        weight_elasticnet * elasticnet_pred_original
    )
except NameError:
    ensemble_pred = (
        weight_xgb * xgb_pred_original + weight_catboost * catboost_pred_original
    )""")

    with open(fname, "w") as f:
        f.write(content)
