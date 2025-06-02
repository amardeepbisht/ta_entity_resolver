import logging
from pyspark.sql import DataFrame
from spark_matcher.matcher import Matcher
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)

def run_ml_spark_matcher(df: DataFrame, config: dict) -> DataFrame:
    """
    Runs fuzzy ML-based matching using spark-matcher with optional blocking.

    Args:
        df (DataFrame): Preprocessed PySpark DataFrame.
        config (dict): Config containing matching columns and thresholds.

    Returns:
        DataFrame: Record pairs with ml_score and record_ids.
    """
    logger.info("Starting ML-based Spark Matcher...")

    match_cfg = config.get("fields_to_match", [])
    logger.debug(f"fields_to_match config: {match_cfg}")  # Debugging line to check config structure
    col_names = [col["name"] for col in match_cfg if col.get("techniques", {}).get("fuzzy", False)]
    
    if not col_names:
        logger.error("No fuzzy match columns found in config.")
        raise ValueError("No fuzzy match columns found in config.")

    logger.info(f"Fuzzy matching will be applied on columns: {col_names}")

    # Optional: Enable blocking rule
    blocking_rule = None
    # Example: Hybrid blocking can be added if needed
    # blocking_rule = HybridVendorBlocking(col_names[0])
    if blocking_rule:
        logger.info("Blocking rule enabled.")
    else:
        logger.info("No blocking rule applied.")

    checkpoint_dir = config.get("checkpoint_dir", "dbfs:/tmp/spark_matcher_checkpoints/")
    logger.info(f"Checkpoint directory set to: {checkpoint_dir}")

    matcher = Matcher(
        spark=df.sql_ctx.sparkSession,
        col_names=col_names,
        blocking_rules=[blocking_rule] if blocking_rule else [],
        checkpoint_dir=checkpoint_dir
    )

    logger.info("Fitting matcher model...")
    matcher.fit(df)

    logger.info("Running prediction...")
    result = matcher.predict(df, df)

    record_id_col = config.get("record_id_column", "record_id")
    logger.info(f"Selecting final columns with record_id: {record_id_col}")

    result = result.select(
        col(f"{record_id_col}_1").alias("record1_id"),
        col(f"{record_id_col}_2").alias("record2_id"),
        col("Score").alias("ml_score")
    )

    logger.info("ML Spark Matcher completed successfully.")
    return result
