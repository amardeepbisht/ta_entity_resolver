# main.py

import sys
import os
import logging

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import DataFrame as SparkDF
from resolver_config.config_loader import load_config
from data_loader.load_data import load_input_data
from preprocessor.preprocessor import preprocess_data
from matcher_engine.matcher_runner import matcher_runner
from postprocessing.ensembler import ensemble_scores


def main():
    # 1) Load configuration
    config_path = os.path.join(PROJECT_ROOT, "resolver_config", "config.yaml")
    config = load_config(config_path)

    # 2) Configure logging
    level = config.get("logging_level", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting orchestration in main.py")

    # 3) Load input data (Pandas or PySpark)
    inp = config["input"]
    engine = config.get("engine", "pandas")
    df = load_input_data(inp, engine=engine)
    # Handle Pandas vs. Spark DataFrame when logging shape
    try:
        rows, cols = df.shape
    except AttributeError:
        rows = df.count()
        cols = len(df.columns)
    logger.info(f"Loaded input ({inp['path']}) → {rows} rows, {cols} cols")

    # 4) Preprocess (this also saves output/preprocessed_data.csv)
    df_clean = preprocess_data(df, config)
    try:
        pre_rows, _ = df_clean.shape
    except AttributeError:
        pre_rows = df_clean.count()
    logger.info(f"After preprocessing → {pre_rows} rows remain")

    # 5) Run all enabled matchers (e.g. fuzzy)
    results = matcher_runner(df_clean, config, engine=config.get("engine", "pandas"))

    # Convert any Spark DataFrame into Pandas before ensembling:
    for name, df in results.items():
        if isinstance(df, SparkDF):
            results[name] = df.toPandas()

    # 6) Ensemble all matcher outputs into a final score
    final_df = ensemble_scores(results, config)

    # 7) Save the final ensembled pairs (use config["output"]["path"])
    out_pairs = config["output"]["resolved_pairs_path"]
    os.makedirs(os.path.dirname(out_pairs), exist_ok=True)
    final_df.to_csv(out_pairs, index=False)
    logger.info(f"Saved final ensembled pairs to {out_pairs} ({len(final_df)} rows)")

    # 8) Write a simple report (use config["output"]["report"])
    report_path = config["output"]["report_path"]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total final pairs: {len(final_df)}\n\n")
        f.write("All matched pairs (with all columns):\n")
        f.write(final_df.to_string(index=False))
    logger.info(f"Saved full summary report to {report_path}")


if __name__ == "__main__":
    main()
