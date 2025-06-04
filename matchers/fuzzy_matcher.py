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
    #logger.info("Running fuzzy matcher...")

    record_id_col = config.get("record_id_column", "record_id")
    fields = config.get("fields_to_match", [])
    string_conf = config.get("matching_techniques_config", {}).get("string_similarity", {})
    default_threshold = string_conf.get("default_threshold", 0.85)

    # --- Step 1: Read boolean flags and filter enabled algorithms ---
    algo_flags = string_conf.get("algorithms", {})
    enabled_algorithms = [name for name, enabled in algo_flags.items() if enabled]

    if not enabled_algorithms:
        raise ValueError("No string-similarity algorithms enabled in config")

    # Prepare match fields with thresholds and weights (unchanged)
    match_fields = []
    for f in fields:
        if f.get("techniques", {}).get("fuzzy", False):
            col_name = f["name"]
            field_thresh = f.get("similarity_threshold", default_threshold)
            field_weight = f.get("weight", 1.0)
            match_fields.append((col_name, field_thresh, field_weight))

    if not match_fields:
        raise ValueError("No fields marked for fuzzy matching in config")

    logger.info(f"Enabled algorithms: {enabled_algorithms}")
    logger.info(f"Matching fields: {match_fields}")

    # Convert DataFrame to list of dicts once
    records = df.to_dict(orient="records")

    # For collecting final results across all algorithms (for return)
    combined_results = []

    # --- Step 2: Loop over each enabled algorithm and build a separate result set ---
    for algorithm_name in enabled_algorithms:
        similarity_fn = ALGORITHM_MAP.get(algorithm_name)
        if similarity_fn is None:
            logger.warning(f"Algorithm '{algorithm_name}' is not in ALGORITHM_MAP; skipping.")
            continue

        logger.info(f"Running algorithm '{algorithm_name}'")
        pairs = []
        total_comparisons = 0
        matched_pairs = 0
        below_threshold = 0

        # Loop through every pair of records
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

                logger.debug(
                    f"[{algorithm_name}] {col_name}: '{val1}' vs '{val2}' → "
                    f"score={score:.4f}, weight={weight}, threshold={thresh}"
                )

            avg_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
            avg_threshold = sum(thresholds) / len(thresholds)

            if avg_score >= avg_threshold:
                matched_pairs += 1
                pairs.append({
                    "record1_id": rec1[record_id_col],
                    "record2_id": rec2[record_id_col],
                    f"{algorithm_name}_score": round(avg_score, 4),
                    "score_type": f"{algorithm_name}_score",
                    "source": "fuzzy_matcher"
                })
            else:
                below_threshold += 1
                logger.debug(
                    f"[{algorithm_name}] Pair ({rec1[record_id_col]}, {rec2[record_id_col]}) skipped: "
                    f"avg_score={avg_score:.4f} < avg_threshold={avg_threshold:.4f}"
                )

        logger.info(
            f"[{algorithm_name}] Completed {total_comparisons} comparisons. "
            f"Matches: {matched_pairs}; Below threshold: {below_threshold}"
        )

        # Convert to DataFrame, save to separate CSV
        result_df = pd.DataFrame(pairs)
        try:
            out_base = config.get("output", {}).get("fuzzy_matcher_path", "output/fuzzy_matcher.csv")
            out_dir = os.path.dirname(out_base)
            os.makedirs(out_dir, exist_ok=True)

            fuzzy_csv = os.path.join(out_dir, f"fuzzy_matcher_{algorithm_name}.csv")
            result_df.to_csv(fuzzy_csv, index=False)
            logger.info(f"[{algorithm_name}] Saved fuzzy matcher output to {fuzzy_csv}")
        except Exception as e:
            logger.warning(f"[{algorithm_name}] Could not save fuzzy matcher output: {e}")

        # Keep for combined return (so downstream code can still pick them up if needed)
        combined_results.append(result_df)

    # If you want a single DataFrame combining all algorithm-specific results:
    #   combined_df = pd.concat(combined_results, ignore_index=True) 
    #   return combined_df
    
    return pd.concat(combined_results, ignore_index=True) if combined_results else pd.DataFrame()

