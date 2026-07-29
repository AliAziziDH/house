import pandas as pd
import numpy as np

# ============================================
# ORDINAL ENCODING
# ============================================

# Quality mappings (from poor to excellent)
quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

# Basement quality mappings
bsmt_qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

# Basement exposure mappings (from no exposure to good exposure)
bsmt_exposure_map = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

# Basement finish type mappings (from unfinished to good living quarters)
bsmt_fin_map = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

# Functional rating mappings (from severely damaged to typical)
functional_map = {
    'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4,
    'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8
}

# Lot shape mappings (from irregular to regular)
lot_shape_map = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}

# Land contour mappings (from low to level)
land_contour_map = {'Low': 1, 'Bnk': 2, 'HLS': 3, 'Lvl': 4}

# Utilities mappings (from limited to all public)
utilities_map = {'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}

# Land slope mappings (from severe to gentle)
land_slope_map = {'Sev': 1, 'Mod': 2, 'Gtl': 3}

# Define ordinal features and their mappings
ordinal_mappings = {
    'ExterQual': quality_map,
    'ExterCond': quality_map,
    'BsmtQual': bsmt_qual_map,
    'BsmtCond': bsmt_qual_map,
    'HeatingQC': quality_map,
    'KitchenQual': quality_map,
    'FireplaceQu': quality_map,
    'GarageQual': quality_map,
    'GarageCond': quality_map,
    'PoolQC': quality_map,
    'BsmtExposure': bsmt_exposure_map,
    'BsmtFinType1': bsmt_fin_map,
    'BsmtFinType2': bsmt_fin_map,
    'Functional': functional_map,
    'LotShape': lot_shape_map,
    'LandContour': land_contour_map,
    'Utilities': utilities_map,
    'LandSlope': land_slope_map,
}

def apply_ordinal_encoding(X_train, X_test, mappings):
    """
    Apply ordinal encoding to the dataset using predefined mappings.
    Features with 'No ...' values are mapped to 0 automatically.
    
    Args:
        X_train: Training features (DataFrame)
        X_test: Test features (DataFrame)
        mappings: Dictionary of column->mapping pairs
    
    Returns:
        Encoded DataFrames (X_train, X_test)
    """
    print("=" * 60)
    print("APPLYING ORDINAL ENCODING")
    print("=" * 60)
    
    # Copy to avoid modifying original data
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    
    for col, mapping in mappings.items():
        if col in X_train_enc.columns:
            X_train_enc[col] = X_train_enc[col].map(mapping)
            X_test_enc[col] = X_test_enc[col].map(mapping)
            
            # Replace unmapped values (like 'No Garage', 'No Basement') with 0
            X_train_enc[col] = X_train_enc[col].fillna(0)
            X_test_enc[col] = X_test_enc[col].fillna(0)
            
            print(f"   ✓ {col}: mapped successfully")
        else:
            print(f"   ⚠️ Warning: {col} not found in data")
    
    print("\n" + "=" * 60)
    print("ORDINAL ENCODING COMPLETED")
    print("=" * 60)
    print(f"   Final number of features: {X_train_enc.shape[1]}")
    print("   All 'No ...' values have been replaced with 0")
    
    return X_train_enc, X_test_enc

# Example usage (commented out - to be used after preprocess.py)
# X_train, X_test = apply_ordinal_encoding(X_train, X_test, ordinal_mappings)