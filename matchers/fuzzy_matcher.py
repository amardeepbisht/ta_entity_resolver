# matchers/fuzzy_matcher.py

import os
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio
from itertools import combinations
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_fuzzy_matcher(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Runs fuzzy matching on specified columns using RapidFuzz, respecting per-field thresholds.
    
    Args:
        df (pd.DataFrame): Input data (Pandas DataFrame)
        config (dict): Config with fields_to_match and record_id_column

    Returns:
        pd.DataFrame: Matched record pairs with fuzzy_score
    """
    logger.info("Running fuzzy matcher...")

    record_id_col = config.get("record_id_column", "record_id")
    fields = config.get("fields_to_match", [])
    default_threshold = config.get("matching_techniques_config", {}) \
                             .get("string_similarity", {}) \
                             .get("default_threshold", 0.85)

    if not fields:
        raise ValueError("No fields_to_match defined in config")

    # Build lists of (column_name, threshold_for_that_column)
    match_fields = []
    for f in fields:
        if f.get("techniques", {}).get("fuzzy", False):
            col_name = f["name"]
            # Per-field threshold if present
            field_thresh = f.get("techniques", {}).get("similarity_threshold", default_threshold)
            match_fields.append((col_name, field_thresh))

    if not match_fields:
        raise ValueError("No fields marked for fuzzy matching in config")

    logger.info(f"Fuzzy match fields and thresholds: {match_fields}")

    pairs = []
    # Convert DataFrame to list of dicts to ease iteration
    records = df.to_dict(orient="records")

    for rec1, rec2 in combinations(records, 2):
        per_field_scores = []
        per_field_thresholds = []
        for col_name, thresh in match_fields:
            val1 = str(rec1.get(col_name, "") or "").strip().lower()
            val2 = str(rec2.get(col_name, "") or "").strip().lower()

            # Use token_set_ratio for better fuzzy matching
            raw_score = token_set_ratio(val1, val2) / 100.0
            per_field_scores.append(raw_score)
            per_field_thresholds.append(thresh)

            logger.debug(f"{col_name}: '{val1}' <-> '{val2}' = {raw_score:.4f} (threshold {thresh})")

        # Compute average score across all fuzzy fields
        avg_score = sum(per_field_scores) / len(per_field_scores)
        # Compute average threshold across all fields
        avg_threshold = sum(per_field_thresholds) / len(per_field_thresholds)

        if avg_score >= avg_threshold:
            pairs.append({
                "record1_id": rec1[record_id_col],
                "record2_id": rec2[record_id_col],
                "fuzzy_score": round(avg_score, 4)
            })

    result_df = pd.DataFrame(pairs)
    logger.info(f"Fuzzy matcher found {len(result_df)} matched pairs.")

    # ── Save this matcher's output so the ensembler can pick it up ──
    try:
        # Determine the output directory from config["output"]["resolved_pairs_path"]
        out_path = config.get("output", {}).get("resolved_pairs_path", None)
        if out_path:
            out_dir = os.path.dirname(out_path)
        else:
            out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)

        fuzzy_csv = os.path.join(out_dir, "fuzzy_matcher.csv")
        result_df.to_csv(fuzzy_csv, index=False)
        logger.info(f"Saved fuzzy matcher output to {fuzzy_csv}")
    except Exception as e:
        logger.warning(f"Could not save fuzzy matcher output: {e}")

    return result_df
