"""
Module: load_data.py
Purpose: Load input dataset using either Pandas or PySpark.
"""

import os
import pandas as pd

# Optional: Future-ready support for Spark processing
try:
    from pyspark.sql import SparkSession
    spark_available = True
except ImportError:
    spark_available = False


def load_input_data(file_path: str, engine: str = "pandas", file_format: str = "csv", sheet_name: str = None):
    """
    Loads input data using the specified engine and file format.

    Args:
        file_path (str): Path to the file.
        engine (str): 'pandas' or 'pyspark'.
        file_format (str): 'csv', 'excel', or 'delta'.
        sheet_name (str): Optional, used for Excel files.

    Returns:
        DataFrame: Loaded Pandas or PySpark DataFrame.
    """
    #print("Current working directory:", os.getcwd())
    #print("Looking for file:",file_path )
    if not os.path.exists(file_path) and engine == "pandas":
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    if engine == "pandas":
        if file_format == "csv":
            return pd.read_csv(file_path)
        elif file_format == "excel":
            return pd.read_excel(file_path, sheet_name=sheet_name or 0)
        else:
            raise ValueError(f"Pandas does not support format: {file_format}")

    elif engine == "pyspark":
        if not spark_available:
            raise ImportError("PySpark is not installed. Please install it to use the 'spark' engine.")

        spark = SparkSession.builder.appName("EntityResolver").getOrCreate()

        if file_format == "csv":
            return spark.read.option("header", "true").csv(file_path)

        elif file_format == "excel":
            try:
                return spark.read.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .option("dataAddress", f"'{sheet_name or 'Sheet1'}'!A1") \
                    .load(file_path)
            except Exception as e:
                raise RuntimeError("Failed to read Excel with PySpark. Make sure 'com.crealytics.spark-excel' is installed on the cluster.") from e

        elif file_format == "delta":
            return spark.read.format("delta").load(file_path)

        else:
            raise ValueError(f"Unsupported file format for PySpark: {file_format}")

    else:
        raise ValueError(f"Unsupported engine: {engine}")
