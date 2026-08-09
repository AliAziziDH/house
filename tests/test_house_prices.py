"""
Automated Pytest Suite for Small-Dataset House Prices Preprocessing.
"""


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

def test_preprocess_data_shapes(tmp_path):
    """Verify data preprocessing pipeline outputs valid non-null datasets."""
    data_dir = "data"
    X_tr, X_te, y_tr, test_ids = preprocess_house_prices_data(data_dir)
    assert len(X_tr) == 1458 # 1460 - 2 outliers
    assert len(X_te) == 1459

if __name__ == '__main__':
    pytest.main(['-v', __file__])
