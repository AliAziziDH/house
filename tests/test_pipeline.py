import pandas as pd
import os
import numpy as np

def test_data_ingestion():
    # Verify raw files are present [2]
    assert os.path.exists("data/train.csv"), "train.csv is missing!"
    
    # Verify the training matrix matches exactly 1,460 rows and 81 columns [2]
    df = pd.read_csv("data/train.csv")
    assert df.shape == (1460, 81), f"Unexpected data shape: {df.shape}"

def test_preprocessing_transforms():
    from src.preprocess import preprocess_data
    
    # Create a fully populated, valid mock dataframe [2]
    mock_data = pd.DataFrame({
        "Id": [3, 4],
        "GrLivArea": ,
        "ExterQual": ["Ex", "TA"],
        "KitchenQual": ["Gd", "Fa"],
        "Neighborhood": ["CollgCr", "Veenker"],
        "SalePrice": 
    })
    
    # Run your production preprocess function [2]
    processed_df = preprocess_data(mock_data, is_training=True)
    
    # Assert that your ordinal quality map successfully mapped string values to numeric types [2]
    assert pd.api.types.is_numeric_dtype(processed_df["ExterQual"]), "ExterQual was not converted to a numeric type!"
