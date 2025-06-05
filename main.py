# main.py
import json
import sys
import os
import logging
import time
import pandas as pd
from resolver_config.config_loader import load_config  
from postprocessing.llm_validator import (
    load_config   as load_llm_config,
    initialize_llm_client,
    read_jsonl_file,
    call_llm_and_parse_response
)
from utilities.file_utils import write_json_to_csv
from utilities.report_generator import run_report_generation


# -----------------------------------------------
#  File Handler Configuration 
# -----------------------------------------------
# Create root logger and set level
logger = logging.getLogger()  
logger.setLevel(logging.INFO)

# Console handler (prints to stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler: write all logs to output/pipeline.log
log_dir = "output"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "pipeline.log"), mode="w")
file_handler.setLevel(logging.INFO)

# Formatter for both handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the root logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ───────────────────────────────────────────────────────────────────────────────


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
from postprocessing.llm_validator import run_llm_validation

start_time = time.time()

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
    logger.info(f"Loaded input ({inp['path']}) -> {rows} rows, {cols} cols")
    logger.info(f"Input loading completed in {time.time() - start_time:.2f} seconds")

    # 4) Preprocess (this also saves output/preprocessed_data.csv)
    df_clean = preprocess_data(df, config)
    try:
        pre_rows, _ = df_clean.shape
    except AttributeError:
        pre_rows = df_clean.count()
    logger.info(f"After preprocessing -> {pre_rows} rows remain")
    logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # 5) Run all enabled matchers (e.g. fuzzy)
    results = matcher_runner(df_clean, config, engine=config.get("engine", "pandas"))

    # Convert any Spark DataFrame into Pandas before ensembling:
    for name, df in results.items():
        if isinstance(df, SparkDF):
            results[name] = df.toPandas()
    logger.info(f"Matcher execution completed in {time.time() - start_time:.2f} seconds")

    # 6) Ensemble all matcher outputs into a final score
    final_df = ensemble_scores(results, config)

    # -----------------------------------------------
    # 6a) Export JSON-lines for LLM validation
    # -----------------------------------------------
    # Build a lookup: record_id -> full preprocessed row (dict of all fields)
    record_id_col = config.get("record_id_column", "record_id")
    lookup = df_clean.set_index(record_id_col).to_dict(orient="index")

    # Decide where to save the JSONL
    llm_json_path = os.path.join(os.path.dirname(config["output"]["resolved_pairs_path"]),
                                 "for_llm.jsonl")
    os.makedirs(os.path.dirname(llm_json_path), exist_ok=True)

    with open(llm_json_path, "w", encoding="utf-8") as f_json:
        for _, row in final_df.iterrows():
            # Grab the full record details from df_clean
            r1 = lookup.get(row["record1_id"], {})
            r2 = lookup.get(row["record2_id"], {})

            # Collect all scores into a nested dict
            scores = {
                key: float(row[key])
                for key in row.index
                if key.endswith("_score")
            }

            # Metadata just tracks IDs
            metadata = {
                "record1_id": str(row["record1_id"]),
                "record2_id": str(row["record2_id"])
            }

            payload = {
                "record1": r1,
                "record2": r2,
                "scores": scores,
                "metadata": metadata
            }
            f_json.write(json.dumps(payload) + "\n")

    logger.info(f"Saved {len(final_df)} LLM-input JSON lines to {llm_json_path}")
    logger.info(f"Ensembling completed in {time.time() - start_time:.2f} seconds")

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


    # Step: Run LLM Validation (if enabled in config)
    llm_enabled = config.get("postprocessing", {}).get("llm_validation", False)
    if llm_enabled:
        logger.info("Running LLM Validation on final matched pairs...")
        validated_results = run_llm_validation(config)

        # write to output file
        validated_output_path = config["output"].get(
            "llm_validated_results_path", 
            "output/llm_validated_results.jsonl"
        )
        with open(validated_output_path, "w", encoding="utf-8") as f_out:
            for item in validated_results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("LLM validated results (JSON) written to: %s", validated_output_path)

        # Write the validated results to CSV
        validated_output_path_csv = config["output"].get(
            "llm_validated_csv_path", 
            "output/llm_validated_results.csv"
        )
        write_json_to_csv(validated_results, validated_output_path_csv)
        logger.info("LLM validated results (CSV) written to: %s", validated_output_path_csv)

        logger.info("LLM validation completed in %.2f seconds", time.time() - start_time)
    else:
        logger.info("LLM Validation is disabled under postprocessing. Skipping.")

     

    # Step: Generate Summary Report (if enabled in config)
    if config.get("postprocessing", {}).get("report_generation", False):
        from utilities.report_generator import run_report_generation

        logger.info("Running Report Generation based on pipeline log...")
        report_text = run_report_generation(config)
        logger.info("Report Generation completed in %.2f seconds", time.time() - start_time)
        # Optionally print or log the generated report text
        logger.debug("Generated Report:\n%s", report_text)
    else:
        logger.info("Report Generation is disabled in config. Skipping.")



if __name__ == "__main__":
    main()
