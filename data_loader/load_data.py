"""
Module: load_data.py
Purpose: Load input dataset using either Pandas or PySpark.
"""

import os
import pandas as pd
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pyspark.sql import SparkSession
    spark_available = True
except ImportError:
    spark_available = False
    logger.warning("PySpark not available. Only Pandas engine will be usable.")

def load_input_data(file_config: dict, engine: str = "pandas"):
    """
    Loads input data using the specified engine and file format.

    Args:
        file_config (dict): Contains path, format, and optionally sheet_name.
        engine (str): 'pandas' or 'pyspark'

    Returns:
        DataFrame: Loaded Pandas or PySpark DataFrame
    """
    file_path = file_config.get("path")
    file_format = file_config.get("format", "csv").lower()
    sheet_name = file_config.get("sheet_name", None)

    logger.info(f"Loading data from {file_path} using engine: {engine} and format: {file_format}")

    if not os.path.exists(file_path) and engine == "pandas":
        logger.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    if engine == "pandas":
        try:
            if file_format == "csv":
                df = pd.read_csv(file_path, encoding_errors='ignore')
            elif file_format == "excel":
                df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
            else:
                raise ValueError(f"Pandas does not support format: {file_format}")
            logger.info(f"Pandas loaded file successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logger.exception("Error while loading file using Pandas:")
            raise

    elif engine == "pyspark":
        if not spark_available:
            raise ImportError("PySpark is not installed. Please install it to use the 'pyspark' engine.")

        spark = SparkSession.builder.appName("EntityResolver").getOrCreate()

        try:
            if file_format == "csv":
                df = spark.read.option("header", "true").csv(file_path)

            elif file_format == "excel":
                df = spark.read.format("com.crealytics.spark.excel") \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .option("dataAddress", f"'{sheet_name or 'Sheet1'}'!A1") \
                    .load(file_path)

            elif file_format == "delta":
                df = spark.read.format("delta").load(file_path)

            else:
                raise ValueError(f"Unsupported file format for PySpark: {file_format}")

            logger.info(f"PySpark loaded file successfully with schema: {df.schema}")
            return df

        except Exception as e:
            logger.exception("Error while loading file using PySpark:")
            raise

    else:
        raise ValueError(f"Unsupported engine: {engine}")
