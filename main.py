# main.py

import sys
import os
import logging

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

    # 3) Load input data (Pandas)
    inp = config["input"]
    engine = config.get("engine", "pandas")
    df = load_input_data(inp, engine=engine)
    logger.info(f"Loaded input ({inp['path']}) → {df.shape[0]} rows, {df.shape[1]} cols")

    # 4) Preprocess (this also saves output/preprocessed_data.csv)
    df_clean = preprocess_data(df, config)
    logger.info(f"After preprocessing → {df_clean.shape[0]} rows remain")

    # 5) Run all enabled matchers (e.g. fuzzy)
    #    Make sure we assign to `results` exactly, not `rresults` or any other name
    results = matcher_runner(df_clean, config, engine=config.get("engine", "pandas"))

    # 6) Ensemble all matcher outputs into a final score
    final_df = ensemble_scores(results, config)

    # 7) Save the final ensembled pairs
    out_pairs = config["output"]["resolved_pairs_path"]
    os.makedirs(os.path.dirname(out_pairs), exist_ok=True)
    final_df.to_csv(out_pairs, index=False)
    logger.info(f"Saved final ensembled pairs to {out_pairs} ({len(final_df)} rows)")

    # 8) Write a simple report
    report_path = config["output"]["report_path"]
    with open(report_path, "w") as f:
        f.write(f"Total final pairs: {len(final_df)}\n")
        f.write("Top 10 pairs by final_score:\n")
        top10 = final_df.sort_values("final_score", ascending=False).head(10)
        f.write(top10.to_string(index=False))
    logger.info(f"Saved summary report to {report_path}")


if __name__ == "__main__":
    main()
