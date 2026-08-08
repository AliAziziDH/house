import os

import numpy as np
import pandas as pd

def preprocess_data(df, is_training=True):
    """Preprocess DataFrame: fill missing values, engineer features, apply ordinal and one-hot encoding."""
    df = df.copy()

    # 1. Garage features
    garage_cat_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    for col in garage_cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("No Garage")

    for col in ["GarageYrBlt", "GarageCars", "GarageArea"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2. Basement features
    bsmt_cat_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    for col in bsmt_cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("No Basement")

    bsmt_num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]
    for col in bsmt_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3. Masonry veneer features
    if "MasVnrType" in df.columns:
        df["MasVnrType"] = df["MasVnrType"].fillna("None")
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # 4. Optional features
    opt_cols = {
        "Alley": "No Alley",
        "PoolQC": "No Pool",
        "Fence": "No Fence",
        "FireplaceQu": "No Fireplace",
        "MiscFeature": "None",
    }
    for col, val in opt_cols.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # 5. Few missing values
    if "LotFrontage" in df.columns:
        if "Neighborhood" in df.columns:
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median())
            )
        median_val = df["LotFrontage"].median()
        if pd.notna(median_val):
            df["LotFrontage"] = df["LotFrontage"].fillna(median_val)
        else:
            df["LotFrontage"] = df["LotFrontage"].fillna(0)

    if "Electrical" in df.columns:
        mode_elec = df["Electrical"].mode()
        if not mode_elec.empty:
            df["Electrical"] = df["Electrical"].fillna(mode_elec[0])

    # 6. Remaining missing values
    cat_cols_with_missing = [
        "MSZoning",
        "Utilities",
        "Exterior1st",
        "Exterior2nd",
        "KitchenQual",
        "Functional",
        "SaleType",
    ]
    for col in cat_cols_with_missing:
        if col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    bsmt_bath_cols = ["BsmtFullBath", "BsmtHalfBath"]
    for col in bsmt_bath_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 7. Feature engineering
    if all(c in df.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if all(c in df.columns for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]):
        df["TotalPorchSF"] = (
            df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        )

    if all(c in df.columns for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    if all(c in df.columns for c in ["YrSold", "YearBuilt"]):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["IsNew"] = (df["YearBuilt"] == df["YrSold"]).astype(int)

    if all(c in df.columns for c in ["YrSold", "YearRemodAdd"]):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    if all(c in df.columns for c in ["OverallQual", "OverallCond"]):
        df["QualityScore"] = df["OverallQual"] * df["OverallCond"]

    if all(c in df.columns for c in ["YrSold", "GarageYrBlt"]):
        df["GarageAge"] = np.where(
            df["GarageYrBlt"] == 0, 0, df["YrSold"] - df["GarageYrBlt"]
        )

    # 8. Ordinal encoding
    quality_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    bsmt_qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    bsmt_exposure_map = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    bsmt_fin_map = {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    functional_map = {
        "Sal": 1,
        "Sev": 2,
        "Maj2": 3,
        "Maj1": 4,
        "Mod": 5,
        "Min2": 6,
        "Min1": 7,
        "Typ": 8,
    }
    lot_shape_map = {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}
    land_contour_map = {"Low": 1, "Bnk": 2, "HLS": 3, "Lvl": 4}
    utilities_map = {"NoSeWa": 1, "NoSewr": 2, "AllPub": 3}
    land_slope_map = {"Sev": 1, "Mod": 2, "Gtl": 3}

    ordinal_mappings = {
        "ExterQual": quality_map,
        "ExterCond": quality_map,
        "BsmtQual": bsmt_qual_map,
        "BsmtCond": bsmt_qual_map,
        "HeatingQC": quality_map,
        "KitchenQual": quality_map,
        "FireplaceQu": quality_map,
        "GarageQual": quality_map,
        "GarageCond": quality_map,
        "PoolQC": quality_map,
        "BsmtExposure": bsmt_exposure_map,
        "BsmtFinType1": bsmt_fin_map,
        "BsmtFinType2": bsmt_fin_map,
        "Functional": functional_map,
        "LotShape": lot_shape_map,
        "LandContour": land_contour_map,
        "Utilities": utilities_map,
        "LandSlope": land_slope_map,
    }

    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # 9. One-hot encoding
    nominal_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    y_train = train["SalePrice"]
    X_train_df = train.drop(["Id", "SalePrice"], axis=1)
    X_test_df = test.drop(["Id"], axis=1)

    X_train = preprocess_data(X_train_df, is_training=True)
    X_test = preprocess_data(X_test_df, is_training=False)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    os.makedirs("./processed_data", exist_ok=True)
    X_train.to_csv("./processed_data/X_train.csv", index=False)
    X_test.to_csv("./processed_data/X_test.csv", index=False)
    y_train.to_csv("./processed_data/y_train.csv", index=False)

    print("✅ Processed data saved successfully.")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print("\n✅ Preprocessing completed successfully.")
