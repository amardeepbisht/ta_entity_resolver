# ensemble/ensembler.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EnsembleError(Exception):
    pass

def ensemble_scores(match_results: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Combine matcher outputs using configured aggregation logic (weighted_average or max_wins),
    and apply a threshold to determine is_match.
    """
    method = config.get("aggregation_method", "weighted_average")
    threshold = config.get("ensemble_threshold", 0.75)
    matcher_weights = config.get("weights", {})

    logger.info(f"Aggregation method: {method}, Threshold: {threshold}")
    logger.info(f"Matcher Weights: {matcher_weights}")

    all_keys = set()
    score_columns = {}

    # Step 1: Collect all unique (record1_id, record2_id) pairs and matcher scores
    pair_scores: Dict[tuple, Dict[str, float]] = {}

    for matcher_name, df in match_results.items():
        score_col = None
        for col in df.columns:
            if col.endswith("_score"):
                score_col = col
                break
        if not score_col:
            logger.warning(f"Matcher {matcher_name} has no score column ending with '_score'. Skipping.")
            continue

        score_columns[matcher_name] = score_col

        for _, row in df.iterrows():
            id1, id2 = row["record1_id"], row["record2_id"]
            key = tuple(sorted((id1, id2)))
            all_keys.add(key)

            if key not in pair_scores:
                pair_scores[key] = {}

            pair_scores[key][matcher_name] = float(row[score_col])

    logger.info(f"Collected {len(all_keys)} unique record pairs from matcher outputs.")

    # Step 2: Compute final score
    rows = []
    for (id1, id2), scores in pair_scores.items():
        values = []
        weights = []

        for matcher_name, score_col in score_columns.items():
            score = scores.get(matcher_name, None)
            weight = matcher_weights.get(matcher_name, 0.0)
            if score is not None:
                values.append(score)
                weights.append(weight)

        if not values:
            final_score = 0.0
            logger.debug(f"No scores for pair ({id1}, {id2}), setting final_score = 0.0")
        elif method == "weighted_average":
            weighted_sum = np.dot(values, weights)
            total_weight = sum(weights)
            final_score = weighted_sum / total_weight if total_weight else 0.0
        elif method == "max_wins":
            final_score = max(values)
        else:
            raise EnsembleError(f"Unsupported aggregation method: {method}")

        is_match = final_score >= threshold
        logger.debug(f"Pair ({id1}, {id2}) → Final Score: {final_score:.4f}, Is Match: {is_match}")

        row = {
            "record1_id": id1,
            "record2_id": id2,
            "final_score": round(final_score, 4),
            "is_match": is_match
        }

        for matcher_name in score_columns:
            col_name = score_columns[matcher_name]
            row[col_name] = scores.get(matcher_name, np.nan)

        rows.append(row)

    result_df = pd.DataFrame(rows)
    logger.info(f"Generated ensemble output with {len(result_df)} rows.")

    return result_df
