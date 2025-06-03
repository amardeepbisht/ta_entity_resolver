# matchers/ml_spark_matcher.py

import logging
import os
from pyspark.sql import DataFrame
from spark_matcher.matcher import Matcher
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)

def run_ml_spark_matcher(df: DataFrame, config: dict) -> DataFrame:
    """
    Runs fuzzy ML-based matching using spark-matcher with optional blocking.
    """
    logger.info("Starting ML-based Spark Matcher...")

    # 1) Extract the list of fields from config
    match_cfg = config.get("fields_to_match", [])
    logger.debug(f"fields_to_match config: {match_cfg}")

    col_names = [
        col_def["name"]
        for col_def in match_cfg
        if col_def.get("techniques", {}).get("fuzzy", False)
    ]
    
    if not col_names:
        logger.error("No fuzzy match columns found in config.")
        raise ValueError("No fuzzy match columns found in config.")
    
    logger.info(f"Fuzzy (ML) matching will be applied on columns: {col_names}")

    # 2) Blocking rule placeholder
    blocking_rule = None
    logger.info("No blocking rule applied.")

    # 3) Setup checkpoint directory
    raw_ckpt = config.get("matching_techniques_config", {}).get("ml_based_spark_matcher", {}).get("checkpoint_dir", "")
    if not raw_ckpt:
        raw_ckpt = "dbfs:/tmp/spark_matcher_checkpoints/"

    on_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    checkpoint_dir = raw_ckpt.replace("dbfs:", "/tmp") if raw_ckpt.startswith("dbfs:") and not on_databricks else raw_ckpt
    logger.info(f"Checkpoint directory set to: {checkpoint_dir}")

    # 4) Run matcher
    matcher = Matcher(
        spark_session=df.sparkSession,
        col_names=col_names,
        blocking_rules=[blocking_rule] if blocking_rule else [],
        checkpoint_dir=checkpoint_dir
    )

    logger.info("Fitting matcher model...")
    matcher.fit(df)

    logger.info("Running prediction...")
    result = matcher.predict(df, df)

    # 5) Rename columns
    res = (
        result
        .withColumnRenamed("Raw_Vendor_Name_1", "name1")
        .withColumnRenamed("Raw_Vendor_Name_2", "name2")
        .withColumnRenamed("score", "ml_score")
    )

    # 6) Log schema and preview
    logger.info("Result schema: " + str(res.columns))
    res.show(5, truncate=False)

    # 7) Save output
    try:
        csv_path = config["output"]["ml_spark_matcher_path"]
        res.toPandas().to_csv(csv_path, index=False)
        logger.info(f"Saved ML Spark Matcher output to {csv_path}")
    except KeyError:
        logger.warning("No 'ml_spark_matcher_path' found under config['output']. Skipping CSV write.")
    except Exception as e:
        logger.error(f"Failed to write ML Spark Matcher CSV: {e}")

    logger.info("ML Spark Matcher completed successfully.")
    return res
