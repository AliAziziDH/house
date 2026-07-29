import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

# ============================================
# LOAD RAW DATA
# ============================================
print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ============================================
# SEPARATE FEATURES AND TARGET
# ============================================
y_train = train['SalePrice']
X_train = train.drop(['Id', 'SalePrice'], axis=1)
X_test = test.drop(['Id'], axis=1)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")

# ============================================
# 1. GARAGE FEATURES
# ============================================
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_cat_cols:
    X_train[col] = X_train[col].fillna('No Garage')
    X_test[col] = X_test[col].fillna('No Garage')

X_train['GarageYrBlt'] = X_train['GarageYrBlt'].fillna(0)
X_test['GarageYrBlt'] = X_test['GarageYrBlt'].fillna(0)

X_train['GarageCars'] = X_train['GarageCars'].fillna(0)
X_test['GarageCars'] = X_test['GarageCars'].fillna(0)

X_train['GarageArea'] = X_train['GarageArea'].fillna(0)
X_test['GarageArea'] = X_test['GarageArea'].fillna(0)

print("✅ Garage features handled.")

# ============================================
# 2. BASEMENT FEATURES
# ============================================
bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_cat_cols:
    X_train[col] = X_train[col].fillna('No Basement')
    X_test[col] = X_test[col].fillna('No Basement')

bsmt_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
for col in bsmt_num_cols:
    X_train[col] = X_train[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)

print("✅ Basement features handled.")

# ============================================
# 3. MASONRY VENEER FEATURES
# ============================================
X_train['MasVnrType'] = X_train['MasVnrType'].fillna('None')
X_test['MasVnrType'] = X_test['MasVnrType'].fillna('None')

X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)
X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)

print("✅ Masonry veneer features handled.")

# ============================================
# 4. OPTIONAL FEATURES
# ============================================
X_train['Alley'] = X_train['Alley'].fillna('No Alley')
X_test['Alley'] = X_test['Alley'].fillna('No Alley')

X_train['PoolQC'] = X_train['PoolQC'].fillna('No Pool')
X_test['PoolQC'] = X_test['PoolQC'].fillna('No Pool')

X_train['Fence'] = X_train['Fence'].fillna('No Fence')
X_test['Fence'] = X_test['Fence'].fillna('No Fence')

X_train['FireplaceQu'] = X_train['FireplaceQu'].fillna('No Fireplace')
X_test['FireplaceQu'] = X_test['FireplaceQu'].fillna('No Fireplace')

X_train['MiscFeature'] = X_train['MiscFeature'].fillna('None')
X_test['MiscFeature'] = X_test['MiscFeature'].fillna('None')

print("✅ Optional features handled.")

# ============================================
# 5. FEW MISSING VALUES
# ============================================
X_train['LotFrontage'] = X_train.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
X_test['LotFrontage'] = X_test.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

most_frequent_electrical = X_train['Electrical'].mode()[0]
X_train['Electrical'] = X_train['Electrical'].fillna(most_frequent_electrical)
X_test['Electrical'] = X_test['Electrical'].fillna(most_frequent_electrical)

print("✅ Features with few missing values handled.")

# ============================================
# 6. REMAINING MISSING VALUES IN TEST
# ============================================
cat_cols_with_missing = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 
                         'KitchenQual', 'Functional', 'SaleType']

for col in cat_cols_with_missing:
    mode_value = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)

bsmt_bath_cols = ['BsmtFullBath', 'BsmtHalfBath']
for col in bsmt_bath_cols:
    X_train[col] = X_train[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)

print("✅ Remaining missing values in test handled.")

# ============================================
# 7. FEATURE ENGINEERING
# ============================================
X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
X_test['TotalSF'] = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']

X_train['TotalPorchSF'] = (X_train['OpenPorchSF'] + X_train['EnclosedPorch'] + 
                           X_train['3SsnPorch'] + X_train['ScreenPorch'])
X_test['TotalPorchSF'] = (X_test['OpenPorchSF'] + X_test['EnclosedPorch'] + 
                          X_test['3SsnPorch'] + X_test['ScreenPorch'])

X_train['TotalBathrooms'] = (X_train['FullBath'] + 0.5 * X_train['HalfBath'] +
                             X_train['BsmtFullBath'] + 0.5 * X_train['BsmtHalfBath'])
X_test['TotalBathrooms'] = (X_test['FullBath'] + 0.5 * X_test['HalfBath'] +
                            X_test['BsmtFullBath'] + 0.5 * X_test['BsmtHalfBath'])

X_train['HouseAge'] = X_train['YrSold'] - X_train['YearBuilt']
X_test['HouseAge'] = X_test['YrSold'] - X_test['YearBuilt']

X_train['RemodAge'] = X_train['YrSold'] - X_train['YearRemodAdd']
X_test['RemodAge'] = X_test['YrSold'] - X_test['YearRemodAdd']

X_train['IsNew'] = (X_train['YearBuilt'] == X_train['YrSold']).astype(int)
X_test['IsNew'] = (X_test['YearBuilt'] == X_test['YrSold']).astype(int)

X_train['QualityScore'] = X_train['OverallQual'] * X_train['OverallCond']
X_test['QualityScore'] = X_test['OverallQual'] * X_test['OverallCond']

X_train['GarageAge'] = np.where(X_train['GarageYrBlt'] == 0, 0, X_train['YrSold'] - X_train['GarageYrBlt'])
X_test['GarageAge'] = np.where(X_test['GarageYrBlt'] == 0, 0, X_test['YrSold'] - X_test['GarageYrBlt'])

print("✅ Feature engineering completed.")
print(f"   New features added: TotalSF, TotalPorchSF, TotalBathrooms, HouseAge, RemodAge, IsNew, QualityScore, GarageAge")

# ============================================
# 8. ORDINAL ENCODING
# ============================================
print("\n" + "=" * 60)
print("APPLYING ORDINAL ENCODING")
print("=" * 60)

quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmt_qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
bsmt_exposure_map = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmt_fin_map = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
functional_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
lot_shape_map = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
land_contour_map = {'Low': 1, 'Bnk': 2, 'HLS': 3, 'Lvl': 4}
utilities_map = {'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}
land_slope_map = {'Sev': 1, 'Mod': 2, 'Gtl': 3}

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

# Save raw data for CatBoost
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

for col, mapping in ordinal_mappings.items():
    if col in X_train.columns:
        X_train[col] = X_train[col].map(mapping).fillna(0)
        X_test[col] = X_test[col].map(mapping).fillna(0)

print("✅ Ordinal encoding completed.")

# ============================================
# 9. ONE-HOT ENCODING
# ============================================
print("\n" + "=" * 60)
print("APPLYING ONE-HOT ENCODING")
print("=" * 60)

nominal_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"   Nominal features to encode: {len(nominal_cols)}")

X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

# Align test columns with train columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print(f"   Final number of features after one-hot encoding: {X_train.shape[1]}")
print("✅ One-hot encoding completed.")

# ============================================
# 10. SAVE PROCESSED DATA
# ============================================
print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

os.makedirs('./processed_data', exist_ok=True)

X_train.to_csv('./processed_data/X_train.csv', index=False)
X_test.to_csv('./processed_data/X_test.csv', index=False)
y_train.to_csv('./processed_data/y_train.csv', index=False)

# Save raw data for CatBoost
X_train_raw.to_csv('./processed_data/X_train_raw.csv', index=False)
X_test_raw.to_csv('./processed_data/X_test_raw.csv', index=False)

print("✅ Processed data saved successfully.")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train: {y_train.shape}")
print("\n✅ Preprocessing completed successfully.")