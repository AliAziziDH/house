import os
import numpy as np
import pandas as pd

from src.preprocess import preprocess_data


def test_data_ingestion():
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv")
        assert df.shape == (1460, 81), f"Unexpected data shape: {df.shape}"


def test_preprocessing_transforms():
    # Create a fully populated, valid mock dataframe
    mock_data = pd.DataFrame({
        "Id": [1, 2],
        "GrLivArea": [1710, 1262],
        "ExterQual": ["Ex", "TA"],
        "KitchenQual": ["Gd", "Fa"],
        "Neighborhood": ["CollgCr", "Veenker"],
        "SalePrice": [208500, 181500]
    })

    # Run production preprocess function
    processed_df = preprocess_data(mock_data, is_training=True)

    # Assert that ordinal quality map successfully mapped string values to numeric types
    assert pd.api.types.is_numeric_dtype(processed_df["ExterQual"]), "ExterQual was not converted to a numeric type!"
    assert pd.api.types.is_numeric_dtype(processed_df["KitchenQual"]), "KitchenQual was not converted to a numeric type!"
    assert processed_df["ExterQual"].iloc[0] == 5
    assert processed_df["ExterQual"].iloc[1] == 3
