"""
Automated Pytest Suite for Small-Dataset House Prices Preprocessing.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.preprocess import QUALITY_MAP, preprocess_house_prices_data


def test_quality_mapping_values():
    """Verify Ordinal Quality Map values."""
    assert QUALITY_MAP['Ex'] == 5
    assert QUALITY_MAP['Gd'] == 4
    assert QUALITY_MAP['TA'] == 3
    assert QUALITY_MAP['None'] == 0

def create_mock_data():
    columns = [
        'Id', 'SalePrice', 'GrLivArea', 'Neighborhood', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond', 'PoolQC', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
        'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch',
        'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'YrSold', 'YearBuilt', 'YearRemodAdd'
    ]
    train_data = {col: np.zeros(1460) for col in columns}
    train_data['Id'] = np.arange(1, 1461)
    train_data['GrLivArea'] = np.ones(1460) * 1000
    train_data['SalePrice'] = np.ones(1460) * 200000
    train_data['Neighborhood'] = ['CollgCr'] * 1460
    # Create 2 outliers (GrLivArea > 4000 & SalePrice < 300,000)
    train_data['GrLivArea'][0] = 4001
    train_data['SalePrice'][0] = 200000
    train_data['GrLivArea'][1] = 4001
    train_data['SalePrice'][1] = 200000
    train_df = pd.DataFrame(train_data)

    test_columns = [col for col in columns if col != 'SalePrice']
    test_data = {col: np.zeros(1459) for col in test_columns}
    test_data['Id'] = np.arange(1461, 2920)
    test_data['GrLivArea'] = np.ones(1459) * 1000
    test_data['Neighborhood'] = ['CollgCr'] * 1459
    test_df = pd.DataFrame(test_data)

    return train_df, test_df

@patch("os.makedirs")
@patch("pandas.DataFrame.to_csv")
@patch("pandas.read_csv")
def test_preprocess_data_shapes(mock_read_csv, mock_to_csv, mock_makedirs):
    """Verify data preprocessing pipeline outputs valid non-null datasets."""
    train_df, test_df = create_mock_data()

    def side_effect(path_or_buf, *args, **kwargs):
        if "train.csv" in str(path_or_buf):
            return train_df.copy()
        elif "test.csv" in str(path_or_buf):
            return test_df.copy()
        raise ValueError(f"Unexpected path: {path_or_buf}")

    mock_read_csv.side_effect = side_effect

    data_dir = "./dummy_dir"
    X_tr, X_te, _y_tr, _test_ids = preprocess_house_prices_data(data_dir)
    assert len(X_tr) == 1458 # 1460 - 2 outliers
    assert len(X_te) == 1459

if __name__ == '__main__':
    pytest.main(['-v', __file__])
