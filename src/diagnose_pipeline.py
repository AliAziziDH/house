import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.preprocess import preprocess_data


def diagnose():
    print("=" * 60)
    print("PIPELINE DIAGNOSTICS")
    print("=" * 60)

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Raw data files missing in data/")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Raw Train shape: {train.shape}")
    print(f"Raw Test shape:  {test.shape}")

    X_train_raw = train.drop(["Id", "SalePrice"], axis=1)
    X_test_raw = test.drop(["Id"], axis=1)

    X_train = preprocess_data(X_train_raw, is_training=True)
    X_test = preprocess_data(X_test_raw, is_training=False)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(f"Processed Train shape: {X_train.shape}")
    print(f"Processed Test shape:  {X_test.shape}")

    train_nans = X_train.isna().sum().sum()
    test_nans = X_test.isna().sum().sum()

    print(f"NaNs in Processed Train: {train_nans}")
    print(f"NaNs in Processed Test:  {test_nans}")

    if train_nans == 0 and test_nans == 0:
        print("✅ Pipeline diagnostics passed: No missing values found.")
    else:
        print("⚠️ Warning: Missing values remain in processed data.")


if __name__ == "__main__":
    diagnose()
