# matchers/preprocessor/preprocessor.py

import os
import pandas as pd
import uuid
import logging

# Optional: Spark compatibility
try:
    from pyspark.sql import DataFrame as SparkDF
    from pyspark.sql.functions import col, lower, trim, monotonically_increasing_id, when
    spark_available = True
except ImportError:
    spark_available = False

logger = logging.getLogger(__name__)

def preprocess_data(df, config: dict):
    """
    Preprocess input data (Pandas or PySpark):
    - Adds record_id if missing
    - Cleans match columns (trim, lowercase)
    - Drops rows with nulls or blanks in match columns
    - Saves the cleaned DataFrame to output/preprocessed_data.csv

    Args:
        df (pd.DataFrame or Spark DataFrame): Input data
        config (dict): Loaded config

    Returns:
        Cleaned DataFrame (same type as input)
    """
    record_id_col = config.get("record_id_column", "record_id")
    match_columns = [col_cfg["name"] for col_cfg in config.get("fields_to_match", [])]

    if isinstance(df, pd.DataFrame):
        # Pandas processing
        # 1. Add record_id column if missing (use sequential integers)
        if record_id_col not in df.columns:
            df[record_id_col] = range(1, len(df) + 1)

        # 2. Clean each match column      
        for col_name in match_columns:
            if col_name not in df.columns:
                raise ValueError(f"Configured match column '{col_name}' not found in input data.")
            df[col_name] = df[col_name].apply(
                lambda x: str(x).strip().lower() if pd.notnull(x) and str(x).strip().lower() not in ["", "nan", "none"] else None
            )

        # 3. Drop rows where any match column is null or blank
        df = df.dropna(subset=match_columns)
        df.reset_index(drop=True, inplace=True)

        # 4. Save preprocessed DataFrame
        try:
            out_path = config.get("output", {}).get("resolved_pairs_path", None)
            if out_path:
                out_dir = os.path.dirname(out_path)
            else:
                out_dir = "output"
            os.makedirs(out_dir, exist_ok=True)
            preproc_csv = os.path.join(out_dir, "preprocessed_data.csv")
            df.to_csv(preproc_csv, index=False)
            logger.info(f"Saved preprocessed data to {preproc_csv}")
        except Exception as e:
            logger.warning(f"Could not save preprocessed data: {e}")

        return df

    elif spark_available and isinstance(df, SparkDF):
        # Spark processing
        spark_cols = df.columns
        for col_name in match_columns:
            if col_name not in spark_cols:
                raise ValueError(f"Configured match column '{col_name}' not found in Spark DataFrame.")

        # Add record_id if missing
        if record_id_col not in spark_cols:
            df = df.withColumn(record_id_col, monotonically_increasing_id())

        # Clean columns (lower, trim, and nullify empty, 'nan', 'none')
        for col_name in match_columns:
            df = df.withColumn(
                col_name,
                when(
                    trim(lower(col(col_name))).isin("", "nan", "none"),
                    None
                ).otherwise(trim(lower(col(col_name))))
            )

        # Drop rows with nulls
        df = df.na.drop(subset=match_columns)

        # Note: saving Spark DataFrame is not implemented here; adapt if needed
        return df

    else:
        raise TypeError("Unsupported input type for preprocessing (must be Pandas or PySpark DataFrame)")
