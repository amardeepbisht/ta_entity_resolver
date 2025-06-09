# main.py
import json
import sys
import os
import logging
import time
import pandas as pd
from ta_entity_resolver.resolver_config.config_loader import load_config
from ta_entity_resolver.postprocessing.llm_validator import (
    load_config as load_llm_config,
    initialize_llm_client,
    read_jsonl_file,
    call_llm_and_parse_response
)
from ta_entity_resolver.utilities.file_utils import write_json_to_csv
from ta_entity_resolver.utilities.report_generator import run_report_generation
from pyspark.sql import DataFrame as SparkDF
from ta_entity_resolver.data_loader.load_data import load_input_data
from ta_entity_resolver.preprocessor.preprocessor import preprocess_data
from ta_entity_resolver.matcher_engine.matcher_runner import matcher_runner
from ta_entity_resolver.postprocessing.ensembler import ensemble_scores
from ta_entity_resolver.postprocessing.llm_validator import run_llm_validation

# -----------------------------------------------
# Module‐level logger and console handler
# -----------------------------------------------
logger = logging.getLogger("ta_entity_resolver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Timing
start_time = time.time()

# Ensure project root is on sys.path for local script runs
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # 1) Load configuration
    config_path = os.path.join(PROJECT_ROOT, "resolver_config", "config.yaml")
    config = load_config(config_path)

    # 1a) Configure log level from config
    level = config.get("logging_level", "INFO").upper()
    logger.setLevel(level)
    console_handler.setLevel(level)

    # 1b) Configure file logging now that config loaded
    pipeline_log = config["output"].get(
        "pipeline_log_path",
        os.path.join(PROJECT_ROOT, "ta_entity_resolver", "output", "pipeline.log")
    )
    log_dir = os.path.dirname(pipeline_log)
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(pipeline_log, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to console and file: {pipeline_log}")

    # 2) Start orchestration
    logger.info("Starting TA Entity Resolver pipeline...")

    # 3) Load input data
    inp = config["input"]
    engine = config.get("engine", "pandas")
    df = load_input_data(inp, engine=engine)
    try:
        rows, cols = df.shape
    except AttributeError:
        rows = df.count()
        cols = len(df.columns)
    logger.info(f"Loaded input ({inp['path']}) -> {rows} rows, {cols} columns")
    logger.info(f"Data load completed in {time.time() - start_time:.2f}s")

    # 4) Preprocess data
    df_clean = preprocess_data(df, config)
    try:
        pre_rows, _ = df_clean.shape
    except AttributeError:
        pre_rows = df_clean.count()
    logger.info(f"After preprocessing: {pre_rows} records remain")
    logger.info(f"Preprocessing completed in {time.time() - start_time:.2f}s")

    # 5) Run matchers
    results = matcher_runner(df_clean, config, engine=engine)
    for name, dfm in results.items():
        if isinstance(dfm, SparkDF):
            results[name] = dfm.toPandas()
    logger.info(f"Matcher execution completed in {time.time() - start_time:.2f}s")

    # 6) Ensemble scores
    final_df = ensemble_scores(results, config)
    logger.info(f"Ensembled into {len(final_df)} record pairs")

    # 6a) Export JSON-lines for LLM validation
    record_id_col = config.get("record_id_column", "record_id")
    if isinstance(df_clean, SparkDF):
        lookup = {row[record_id_col]: row.asDict() for row in df_clean.collect()}
    else:
        lookup = df_clean.set_index(record_id_col).to_dict(orient="index")

    llm_json_path = os.path.join(
        os.path.dirname(config["output"]["resolved_pairs_path"]),
        "for_llm.jsonl"
    )
    os.makedirs(os.path.dirname(llm_json_path), exist_ok=True)
    with open(llm_json_path, "w", encoding="utf-8") as f:
        for _, row in final_df.iterrows():
            r1 = lookup.get(row["record1_id"], {})
            r2 = lookup.get(row["record2_id"], {})
            scores = {k: float(row[k]) for k in row.index if k.endswith("_score")}
            payload = {"record1": r1, "record2": r2, "scores": scores, "metadata": {"record1_id": str(row["record1_id"]), "record2_id": str(row["record2_id"])}}
            f.write(json.dumps(payload) + "\n")
    logger.info(f"Saved {len(final_df)} LLM-input JSON lines to {llm_json_path}")

    # 7) Save final pairs
    out_pairs = config["output"]["resolved_pairs_path"]
    os.makedirs(os.path.dirname(out_pairs), exist_ok=True)
    final_df.to_csv(out_pairs, index=False)
    logger.info(f"Saved final pairs to {out_pairs}")

    # 8) Write summary report
    report_path = config["output"]["report_path"]
    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write(f"Total final pairs: {len(final_df)}\n\n")
        rpt.write(final_df.to_string(index=False))
    logger.info(f"Saved summary report to {report_path}")

    # 9) Optional LLM validation
    if config.get("postprocessing", {}).get("llm_validation", False):
        logger.info("Running LLM validation...")
        validated = run_llm_validation(config)
        csv_path = config["output"]["llm_validated_csv_path"]
        write_json_to_csv(validated, csv_path)
        logger.info(f"LLM validation results saved to {csv_path}")

    # 10) Optional additional report generation
    if config.get("postprocessing", {}).get("report_generation", False):
        logger.info("Running additional report generation...")
        run_report_generation(config)
        logger.info("Additional report generation complete")


if __name__ == "__main__":
    main()
