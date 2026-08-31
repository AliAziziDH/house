
import sys

import pandas as pd

SUBMISSION_PATH = "submissions/submission_ensemble_oof.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {SUBMISSION_PATH} ...")
try:
    df = pd.read_csv(SUBMISSION_PATH)
except FileNotFoundError:
    print(f"❌ Error: File {SUBMISSION_PATH} not found.")
    sys.exit(1)

# Shape and completeness checks
assert len(df) == 1459, f"❌ Shape error: expected 1459 rows, got {len(df)}"
assert df["Id"].min() == 1461 and df["Id"].max() == 2919, "❌ ID error: IDs do not match test IDs [1461, 2919]."
assert df["SalePrice"].isnull().sum() == 0, "❌ NaN values detected in SalePrice."

# Boundary checks
assert df["SalePrice"].min() >= 42000, f"❌ Boundary error: minimum price {df['SalePrice'].min()} < 42000"
assert df["SalePrice"].max() <= 525000, f"❌ Boundary error: maximum price {df['SalePrice'].max()} > 525000"

print("✅ Shape, ID matching, NaN checks, and boundary constraints passed.")

print("\n" + "=" * 55)
print("  SUBMISSION QA REPORT")
print("=" * 55)
print(f"  Rows validated      : {len(df)}")
print(f"  Missing values      : {df['SalePrice'].isnull().sum()}")
print("  Boundary [42k, 525k]: ✅ PASS")

# ── Summary Statistics ────────────────────────────────────────────────────────
stats = {
    "count": len(df),
    "mean": df["SalePrice"].mean(),
    "std": df["SalePrice"].std(),
    "min": df["SalePrice"].min(),
    "25%": df["SalePrice"].quantile(0.25),
    "50%": df["SalePrice"].median(),
    "75%": df["SalePrice"].quantile(0.75),
    "max": df["SalePrice"].max(),
}

print("\n  Summary Statistics (SalePrice)\n")
for k, v in stats.items():
    if k == "count":
        print(f"  {k:<10}: {v}")
    else:
        print(f"  {k:<10}: ${v:,.0f}")
print("=" * 55 + "\n")
