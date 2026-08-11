import pandas as pd
import numpy as np

for fname in ["src/ensemble.py", "src/ensemble_oof.py"]:
    with open(fname, "r") as f:
        content = f.read()

    # Fix 1: Properly normalize the fallback weights so they sum to 1.0!
    # "Because the weights sum to roughly 0.3334 instead of 1.0... the pipeline will drastically underestimate all house prices"
    content = content.replace("""except NameError:
    ensemble_pred = (
        weight_xgb * xgb_pred_original + weight_catboost * catboost_pred_original
    )""", """except NameError:
    # Normalize weights to sum to 1.0
    total = weight_xgb + weight_catboost
    ensemble_pred = (
        (weight_xgb / total) * xgb_pred_original + (weight_catboost / total) * catboost_pred_original
    )""")

    with open(fname, "w") as f:
        f.write(content)
