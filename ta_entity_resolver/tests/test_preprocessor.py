import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
from preprocessor.preprocessor import preprocess_data

def test_preprocess_basic_cleaning():
    raw_df = pd.DataFrame({
        "Raw_Vendor_Name": ["  Uber ", "uber tech", None, " ", "Uber.com"]
    })

    config = {
        "record_id_column": "record_id",
        "match_columns": [{"name": "Raw_Vendor_Name", "technique": "fuzzy", "weight": 1}]
    }

    cleaned_df = preprocess_data(raw_df, config)

    assert "record_id" in cleaned_df.columns
    assert cleaned_df.shape[0] == 3  # Two null/blank rows removed
    assert cleaned_df["Raw_Vendor_Name"].tolist() == ["uber", "uber tech", "uber.com"]
