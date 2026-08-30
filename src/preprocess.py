import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "BSMT_EXPOSURE_MAP",
    "BSMT_FIN_MAP",
    "BSMT_QUAL_MAP",
    "FUNCTIONAL_MAP",
    "LAND_CONTOUR_MAP",
    "LAND_SLOPE_MAP",
    "LOT_SHAPE_MAP",
    "QUALITY_MAP",
    "UTILITIES_MAP",
    "AmesDataTransformer",
    "preprocess_data",
]

# Ordinal mappings
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
BSMT_QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
BSMT_EXPOSURE_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
BSMT_FIN_MAP = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
FUNCTIONAL_MAP = {
    "Typ": 8,
    "Min1": 7,
    "Min2": 6,
    "Mod": 5,
    "Maj1": 4,
    "Maj2": 3,
    "Sev": 2,
    "Sal": 1,
}
LOT_SHAPE_MAP = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
LAND_CONTOUR_MAP = {"Lvl": 4, "HLS": 3, "Bnk": 2, "Low": 1}
UTILITIES_MAP = {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1}
LAND_SLOPE_MAP = {"Gtl": 3, "Mod": 2, "Sev": 1}


class AmesDataTransformer(BaseEstimator, TransformerMixin):
    """
    Stateful Scikit-Learn transformer for Ames Housing data.
    Fits all statistics (medians, modes, target ranks, one-hot schema) strictly on training data
    and applies them without data leakage during transform.
    """

    def __init__(self):
        self.lot_frontage_neighborhood_medians_ = {}
        self.lot_frontage_global_median_ = 0.0
        self.categorical_modes_ = {}
        self.neighborhood_target_ranks_ = {}
        self.global_neighborhood_rank_ = 0.0
        self.feature_columns_ = []

    def fit(self, X, y=None):
        X = X.copy()

        # 1. LotFrontage statistics
        if "LotFrontage" in X.columns:
            if "Neighborhood" in X.columns:
                self.lot_frontage_neighborhood_medians_ = (
                    X.groupby("Neighborhood")["LotFrontage"].median().to_dict()
                )
            med_val = X["LotFrontage"].median()
            self.lot_frontage_global_median_ = float(med_val) if pd.notna(med_val) else 0.0

        # 2. Categorical modes
        cat_cols_with_missing = [
            "Electrical",
            "MSZoning",
            "Utilities",
            "Exterior1st",
            "Exterior2nd",
            "KitchenQual",
            "Functional",
            "SaleType",
        ]
        for col in cat_cols_with_missing:
            if col in X.columns:
                mode_val = X[col].mode()
                if not mode_val.empty:
                    self.categorical_modes_[col] = mode_val[0]

        # 3. Neighborhood Target Ranking (if y is provided)
        if y is not None and "Neighborhood" in X.columns:
            df_target = pd.DataFrame({"Neighborhood": X["Neighborhood"], "Target": y})
            neigh_medians = df_target.groupby("Neighborhood")["Target"].median()
            neigh_ranks = neigh_medians.rank(method="min").to_dict()
            self.neighborhood_target_ranks_ = neigh_ranks
            self.global_neighborhood_rank_ = (
                float(np.median(list(neigh_ranks.values()))) if neigh_ranks else 0.0
            )

        # 4. Transform training data to learn final column schema
        X_trans = self._transform_df(X)
        self.feature_columns_ = X_trans.columns.tolist()

        return self

    def _transform_df(self, df):
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

        # 5. LotFrontage imputation using fitted medians
        if "LotFrontage" in df.columns:
            if "Neighborhood" in df.columns and self.lot_frontage_neighborhood_medians_:
                neigh_series = df["Neighborhood"].map(self.lot_frontage_neighborhood_medians_)
                df["LotFrontage"] = df["LotFrontage"].fillna(neigh_series)
            df["LotFrontage"] = df["LotFrontage"].fillna(self.lot_frontage_global_median_)

        # 6. Categorical mode imputation using fitted modes
        for col, mode_val in self.categorical_modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

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

        # Neighborhood Target Rank feature
        if "Neighborhood" in df.columns and self.neighborhood_target_ranks_:
            mapped_ranks = df["Neighborhood"].map(self.neighborhood_target_ranks_)
            df["NeighborhoodTargetRank"] = mapped_ranks.fillna(self.global_neighborhood_rank_)

        # 8. Ordinal encoding
        ordinal_mappings = {
            "ExterQual": QUALITY_MAP,
            "ExterCond": QUALITY_MAP,
            "BsmtQual": BSMT_QUAL_MAP,
            "BsmtCond": BSMT_QUAL_MAP,
            "HeatingQC": QUALITY_MAP,
            "KitchenQual": QUALITY_MAP,
            "FireplaceQu": QUALITY_MAP,
            "GarageQual": QUALITY_MAP,
            "GarageCond": QUALITY_MAP,
            "PoolQC": QUALITY_MAP,
            "BsmtExposure": BSMT_EXPOSURE_MAP,
            "BsmtFinType1": BSMT_FIN_MAP,
            "BsmtFinType2": BSMT_FIN_MAP,
            "Functional": FUNCTIONAL_MAP,
            "LotShape": LOT_SHAPE_MAP,
            "LandContour": LAND_CONTOUR_MAP,
            "Utilities": UTILITIES_MAP,
            "LandSlope": LAND_SLOPE_MAP,
        }

        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        # 9. One-hot encoding
        nominal_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if nominal_cols:
            df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

        return df

    def transform(self, X):
        X_trans = self._transform_df(X)
        if self.feature_columns_:
            X_trans = X_trans.reindex(columns=self.feature_columns_, fill_value=0)
        return X_trans


def preprocess_data(df, is_training=True):
    """
    Backward-compatible preprocessing function using stateful AmesDataTransformer.
    """
    transformer = AmesDataTransformer()
    if "SalePrice" in df.columns:
        y = df["SalePrice"]
        X = df.drop(columns=["SalePrice"])
    else:
        y = None
        X = df

    transformer.fit(X, y)
    return transformer.transform(X)


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

    transformer = AmesDataTransformer()
    transformer.fit(X_train_df, y_train)

    X_train = transformer.transform(X_train_df)
    X_test = transformer.transform(X_test_df)

    os.makedirs("./processed_data", exist_ok=True)
    X_train.to_csv("./processed_data/X_train.csv", index=False)
    X_test.to_csv("./processed_data/X_test.csv", index=False)
    y_train.to_csv("./processed_data/y_train.csv", index=False)

    print("✅ Processed data saved successfully.")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print("\n✅ Preprocessing completed successfully.")
