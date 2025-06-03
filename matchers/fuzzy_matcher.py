# matchers/fuzzy_matcher.py

import os
import pandas as pd
from itertools import combinations
import logging
from typing import Dict, Any
from rapidfuzz import fuzz, distance

logger = logging.getLogger(__name__)

# --- Mapping of algorithm names to RapidFuzz functions ---
ALGORITHM_MAP = {
    "jaro_winkler": lambda a, b: distance.JaroWinkler.similarity(a, b),
    "levenshtein": lambda a, b: 1 - distance.Levenshtein.normalized_distance(a, b),
    "token_set_ratio": lambda a, b: fuzz.token_set_ratio(a, b) / 100.0
}

def run_fuzzy_matcher(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Running fuzzy matcher...")

    record_id_col = config.get("record_id_column", "record_id")
    fields = config.get("fields_to_match", [])
    string_conf = config.get("matching_techniques_config", {}).get("string_similarity", {})
    default_threshold = string_conf.get("default_threshold", 0.85)
    algorithm_name = string_conf.get("algorithm", "token_set_ratio")
    similarity_fn = ALGORITHM_MAP.get(algorithm_name)

    if similarity_fn is None:
        raise ValueError(f"Unsupported similarity algorithm: {algorithm_name}")

    if not fields:
        raise ValueError("No fields_to_match defined in config")

    # Prepare match fields with thresholds and weights
    match_fields = []
    for f in fields:
        if f.get("techniques", {}).get("fuzzy", False):
            col_name = f["name"]
            field_thresh = f.get("techniques", {}).get("similarity_threshold", default_threshold)
            field_weight = f.get("weight", 1.0)
            match_fields.append((col_name, field_thresh, field_weight))

    if not match_fields:
        raise ValueError("No fields marked for fuzzy matching in config")

    logger.info(f"Using algorithm '{algorithm_name}' with fields: {match_fields}")

    pairs = []
    total_comparisons = 0
    matched_pairs = 0
    below_threshold = 0

    records = df.to_dict(orient="records")
    for rec1, rec2 in combinations(records, 2):
        total_comparisons += 1
        weighted_scores = []
        total_weight = 0
        thresholds = []

        for col_name, thresh, weight in match_fields:
            val1 = str(rec1.get(col_name, "") or "").strip().lower()
            val2 = str(rec2.get(col_name, "") or "").strip().lower()
            score = similarity_fn(val1, val2)

            weighted_scores.append(score * weight)
            thresholds.append(thresh)
            total_weight += weight

            logger.debug(f"{col_name}: '{val1}' vs '{val2}' → score={score:.4f}, weight={weight}, threshold={thresh}")

        avg_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
        avg_threshold = sum(thresholds) / len(thresholds)

        if avg_score >= avg_threshold:
            matched_pairs += 1
            pairs.append({
                "record1_id": rec1[record_id_col],
                "record2_id": rec2[record_id_col],
                "fuzzy_score": round(avg_score, 4),          # ensure fuzzy_score is the only score column
                "score_type": "fuzzy_score",
                "source": "fuzzy_matcher"
            })
        else:
            below_threshold += 1
            logger.debug(
                f"Pair ({rec1[record_id_col]}, {rec2[record_id_col]}) skipped: "
                f"avg_score={avg_score:.4f} < avg_threshold={avg_threshold:.4f}"
            )

    logger.info(f"Completed {total_comparisons} comparisons.")
    logger.info(f"Fuzzy matcher found {matched_pairs} matched pairs; {below_threshold} pairs below threshold.")

    result_df = pd.DataFrame(pairs)
    # At this point, result_df.columns should be:
    # ['record1_id', 'record2_id', 'fuzzy_score', 'score_type', 'source']

    try:
        out_base = config.get("output", {}).get("fuzzy_matcher_path", "output/fuzzy_matcher.csv")
        out_dir = os.path.dirname(out_base)
        os.makedirs(out_dir, exist_ok=True)

        # Build the dynamic output filename
        fuzzy_csv = os.path.join(out_dir, f"fuzzy_matcher_{algorithm_name}.csv")
        result_df.to_csv(fuzzy_csv, index=False)
        logger.info(f"Saved fuzzy matcher output to {fuzzy_csv}")
    except Exception as e:
        logger.warning(f"Could not save fuzzy matcher output: {e}")

    return result_df
