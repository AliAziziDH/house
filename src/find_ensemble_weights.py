"""
Ensemble weight optimisation via SLSQP.

Delegates all solver logic to scripts.run_slsqp.optimize_weights so that
the blending algorithm is defined in exactly one place.

Input OOF files expected in ./processed_data/:
  oof_xgboost.csv   — column: OOF_SalePrice
  oof_catboost.csv  — column: OOF_SalePrice
  oof_linear.csv    — column: OOF_SalePrice  (optional, skipped if absent)

Output:
  submissions/submission_ensemble_rmsle_final.csv
  experiments/slsqp_weights.csv
"""

import os

import numpy as np
import pandas as pd

from scripts.run_slsqp import optimize_weights


# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
OOF_DIR = "./processed_data"
TEST_DATA = "./data/test.csv"
SUBMISSION_DIR = "./submissions"
EXPERIMENTS_DIR = "./experiments"

# Map model name -> OOF file path (add/remove models here)
OOF_FILES = {
    "XGBoost": os.path.join(OOF_DIR, "oof_xgboost.csv"),
    "CatBoost": os.path.join(OOF_DIR, "oof_catboost.csv"),
    "Linear": os.path.join(OOF_DIR, "oof_linear.csv"),
}
OOF_COL = "OOF_SalePrice"


def main():
    print("=" * 60)
    print("LOADING OOF PREDICTIONS & TARGET")
    print("=" * 60)

    y_train = pd.read_csv(os.path.join(OOF_DIR, "y_train.csv")).squeeze()

    # Build oof_predictions DataFrame; skip missing files with a warning
    oof_dict = {}
    for name, path in OOF_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠️  {name}: OOF file not found ({path}), skipping.")
            continue
        df = pd.read_csv(path)
        if OOF_COL not in df.columns:
            raise ValueError(f"{path} missing expected column '{OOF_COL}'")
        oof_dict[name] = df[OOF_COL].values
        print(f"  ✅ {name}: loaded {len(oof_dict[name])} OOF predictions")

    if len(oof_dict) < 2:
        raise RuntimeError("Need at least 2 OOF files to blend.")

    oof_predictions = pd.DataFrame(oof_dict)
    print(f"\nOOF matrix shape: {oof_predictions.shape}")
    print(f"Target shape:     {y_train.shape}")

    # ============================================
    # SLSQP WEIGHT OPTIMISATION
    # ============================================
    print("\n" + "=" * 60)
    print("RUNNING SLSQP BLENDING OPTIMISATION")
    print("=" * 60)

    best_weights, best_rmsle = optimize_weights(
        oof_predictions=oof_predictions,
        y_true=y_train.values,
    )

    print("\n✅ Optimal blending weights:")
    for model, w in best_weights.items():
        print(f"   {model:12s}: {w:.6f}")
    print(f"✅ Best OOF RMSLE: {best_rmsle:.6f}")

    assert abs(sum(best_weights.values()) - 1.0) < 1e-6, (
        f"Weight sum = {sum(best_weights.values()):.8f} — must equal 1.0"
    )
    assert best_rmsle < 0.1180, (
        f"Guardrail breached: OOF RMSLE = {best_rmsle:.6f} >= 0.1180"
        " — regenerate OOF files from tuned models before submitting."
    )

    # ============================================
    # SAVE WEIGHT REPORT
    # ============================================
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    weights_df = pd.DataFrame(
        [{"model": m, "weight": w} for m, w in best_weights.items()]
    )
    weights_df["oof_rmsle"] = best_rmsle
    weights_path = os.path.join(EXPERIMENTS_DIR, "slsqp_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    print(f"\n✅ Weights saved to {weights_path}")

    print("\n" + "=" * 60)
    print("ENSEMBLE WEIGHT OPTIMISATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
