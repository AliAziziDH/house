"""
QA script: cross-validate AutoGluon and XGBoost submission files.

Checks:
  1. Pearson correlation between predicted SalePrice columns (must be > 0.90).
  2. Side-by-side summary statistics for both submissions.

Usage:
    python src/check_predictions.py
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

XGBOOST_PATH  = "submissions/submission_xgboost_rmsle.csv"
AUTOGLUON_PATH = "submissions/submission_autogluon.csv"
CORRELATION_THRESHOLD = 0.90

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {XGBOOST_PATH} ...")
xgb = pd.read_csv(XGBOOST_PATH)

print(f"Loading {AUTOGLUON_PATH} ...")
agl = pd.read_csv(AUTOGLUON_PATH)

# Sanity: both must have 'Id' and 'SalePrice'
for name, df in [("XGBoost", xgb), ("AutoGluon", agl)]:
    assert {"Id", "SalePrice"}.issubset(df.columns), \
        f"{name}: missing required columns. Found: {df.columns.tolist()}"
    assert df["SalePrice"].isnull().sum() == 0, \
        f"{name}: NaN values detected in SalePrice."
    assert (df["SalePrice"] > 0).all(), \
        f"{name}: Non-positive SalePrice values detected."

# Align on Id in case row order differs
merged = xgb.rename(columns={"SalePrice": "xgb"}).merge(
    agl.rename(columns={"SalePrice": "agl"}),
    on="Id", how="inner",
)
assert len(merged) == len(xgb) == len(agl), \
    f"Id mismatch between files: {len(xgb)} vs {len(agl)} rows, {len(merged)} matched."

# ── Pearson Correlation ───────────────────────────────────────────────────────
corr, p_value = pearsonr(merged["xgb"], merged["agl"])

print("\n" + "=" * 55)
print("  SUBMISSION QA REPORT")
print("=" * 55)
print(f"  Rows compared       : {len(merged)}")
print(f"  Pearson Correlation : {corr:.4f}  (p={p_value:.2e})")

if corr >= CORRELATION_THRESHOLD:
    print(f"  Correlation check   : ✅ PASS  (≥ {CORRELATION_THRESHOLD})")
else:
    print(f"  Correlation check   : ❌ FAIL  (< {CORRELATION_THRESHOLD})")
    print("  ⚠️  Structural misalignment detected — investigate target transform.")

# ── Summary Statistics ────────────────────────────────────────────────────────
stats = pd.DataFrame(
    {
        "XGBoost (xgboost_rmsle)": {
            "min":    merged["xgb"].min(),
            "max":    merged["xgb"].max(),
            "median": merged["xgb"].median(),
            "mean":   merged["xgb"].mean(),
            "std":    merged["xgb"].std(),
        },
        "AutoGluon (autogluon)": {
            "min":    merged["agl"].min(),
            "max":    merged["agl"].max(),
            "median": merged["agl"].median(),
            "mean":   merged["agl"].mean(),
            "std":    merged["agl"].std(),
        },
    }
)

print("\n  Side-by-side Summary Statistics (SalePrice)\n")
print(stats.map(lambda x: f"${x:,.0f}").to_string())
print("=" * 55 + "\n")

# Exit with non-zero code if correlation gate fails (CI-friendly)
if corr < CORRELATION_THRESHOLD:
    sys.exit(1)
