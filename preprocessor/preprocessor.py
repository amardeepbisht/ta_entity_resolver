import pandas as pd
import uuid

# Optional: Spark compatibility
try:
    from pyspark.sql import DataFrame as SparkDF
    from pyspark.sql.functions import col, lower, trim, monotonically_increasing_id
    from pyspark.sql.functions import when
    spark_available = True
except ImportError:
    spark_available = False


def preprocess_data(df, config: dict):
    """
    Preprocess input data (Pandas or PySpark):
    - Adds record_id if missing
    - Cleans match columns (trim, lowercase)
    - Drops rows with nulls or blanks in match columns

    Args:
        df (pd.DataFrame or Spark DataFrame): Input data
        config (dict): Loaded config

    Returns:
        Cleaned DataFrame (same type as input)
    """
    record_id_col = config.get("record_id_column", "record_id")
    match_columns = [col_cfg["name"] for col_cfg in config.get("match_columns", [])]

    if isinstance(df, pd.DataFrame):
        # Pandas processing
        # 1. Add record_id column if missing (use sequential integers for easier readability/debugging)
        if record_id_col not in df.columns:
            df[record_id_col] = range(1, len(df) + 1)

        # uuid generation is not used here, but can be uncommented if needed
        # if record_id_col not in df.columns:
        #     df[record_id_col] = [str(uuid.uuid4()) for _ in range(len(df))]

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

        return df

    else:
        raise TypeError("Unsupported input type for preprocessing (must be Pandas or PySpark DataFrame)")
