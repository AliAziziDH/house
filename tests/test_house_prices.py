"""
Automated Pytest Suite for Small-Dataset House Prices Preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocess import preprocess_house_prices_data, QUALITY_MAP

def test_quality_mapping_values():
    """Verify Ordinal Quality Map values."""
    assert QUALITY_MAP['Ex'] == 5
    assert QUALITY_MAP['Gd'] == 4
    assert QUALITY_MAP['TA'] == 3
    assert QUALITY_MAP['None'] == 0

def test_preprocess_data_shapes():
    """Verify data preprocessing pipeline outputs valid non-null datasets."""
    data_dir = "./data"

    # Skip test if actual data files are missing during testing environments
    if not (Path(data_dir) / "train.csv").exists():
        pytest.skip("Kaggle data files missing in ./data")

    X_tr, X_te, y_tr, test_ids = preprocess_house_prices_data(data_dir)
    assert len(X_tr) == 1458 # 1460 - 2 outliers
    assert len(X_te) == 1459
    assert not X_tr.isnull().values.any()
    assert not X_te.isnull().values.any()

if __name__ == '__main__':
    pytest.main(['-v', __file__])
