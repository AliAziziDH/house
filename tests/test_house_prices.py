"""
Automated Pytest Suite for Small-Dataset House Prices Preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.preprocess import preprocess_house_prices_data, QUALITY_MAP
from src.find_ensemble_weights import rmsle_dollars

def test_quality_mapping_values():
    """Verify Ordinal Quality Map values."""
    assert QUALITY_MAP['Ex'] == 5
    assert QUALITY_MAP['Gd'] == 4
    assert QUALITY_MAP['TA'] == 3
    assert QUALITY_MAP['None'] == 0

def test_preprocess_data_shapes():
    """Verify data preprocessing pipeline outputs valid non-null datasets."""
    data_dir = "./data"
    X_tr, X_te, y_tr, test_ids = preprocess_house_prices_data(data_dir)
    assert len(X_tr) == 1458 # 1460 - 2 outliers
    assert len(X_te) == 1459
    assert not X_tr.isnull().values.any()
    assert not X_te.isnull().values.any()


def test_rmsle_dollars_perfect_match():
    """Verify exact match returns RMSLE of 0."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    assert np.isclose(rmsle_dollars(y_true, y_pred), 0.0)

def test_rmsle_dollars_negative_clamping():
    """Verify negative values are correctly clamped to 0.0."""
    y_true = np.array([0.0, 5.0])
    y_pred = np.array([-10.0, 5.0])
    # -10 gets clamped to 0.0. RMSLE should be 0 since both arrays effectively become [0.0, 5.0]
    assert np.isclose(rmsle_dollars(y_true, y_pred), 0.0)

def test_rmsle_dollars_known_values():
    """Verify RMSLE calculation yields expected mathematical results."""
    # log1p(e-1) = 1, log1p(0) = 0
    y_true = np.array([np.exp(1) - 1])
    y_pred = np.array([0.0])
    # log difference is 1 - 0 = 1, squared is 1, mean is 1, sqrt is 1
    assert np.isclose(rmsle_dollars(y_true, y_pred), 1.0)

def test_rmsle_dollars_zero_values():
    """Verify behavior when both inputs are exactly zero."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert np.isclose(rmsle_dollars(y_true, y_pred), 0.0)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
