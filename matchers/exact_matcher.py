# matchers/exact_matcher.py

import os
import pandas as pd
from itertools import combinations
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def run_exact_matcher(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    #ogger.info("Running exact matcher...")

    # Record ID column
    record_id_col = config.get("record_id_column", "record_id")

    # Fields to consider for exact matching
    fields = config.get("fields_to_match", [])
    match_fields = [
        f["name"]
        for f in fields
        if f.get("techniques", {}).get("exact", False)
    ]

    if not match_fields:
        raise ValueError("No fields marked for exact matching in config")

    logger.info(f"Exact-matching on fields: {match_fields}")

    pairs = []
    total_comparisons = 0
    matched_pairs = 0

    records = df.to_dict(orient="records")
    for rec1, rec2 in combinations(records, 2):
        total_comparisons += 1

        # Compare each exact-enabled field
        is_exact = True
        for col_name in match_fields:
            val1 = str(rec1.get(col_name, "") or "").strip().lower()
            val2 = str(rec2.get(col_name, "") or "").strip().lower()
            if val1 != val2:
                is_exact = False
                break

        if is_exact:
            matched_pairs += 1
            pairs.append({
                "record1_id": rec1[record_id_col],
                "record2_id": rec2[record_id_col],
                "exact_score": 1.0,
                "score_type": "exact_score",
                "source": "exact_matcher"
            })

    logger.info(f"Completed {total_comparisons} comparisons.")
    logger.info(f"Exact matcher found {matched_pairs} matched pairs.")

    result_df = pd.DataFrame(pairs)

    # Write CSV
    try:
        out_base = config.get("output", {}).get("exact_matcher_path", "output/exact_matcher.csv")
        out_dir = os.path.dirname(out_base)
        os.makedirs(out_dir, exist_ok=True)

        exact_csv = os.path.join(out_dir, "exact_matcher.csv")
        result_df.to_csv(exact_csv, index=False)
        logger.info(f"Saved exact matcher output to {exact_csv}")
    except Exception as e:
        logger.warning(f"Could not save exact matcher output: {e}")

    return result_df
