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
    Combine individual matcher outputs into a final_score using boss’s logic:
      - Gather only non-zero technique scores for each pair.
      - Apply either weighted_average or max_wins based on config.

    match_results: {
      "exact_matcher": DataFrame with columns record1_id, record2_id, exact_match,
      "normalized_matcher": DataFrame with record1_id, record2_id, normalized_match,
      "phonetic_matcher": DataFrame with record1_id, record2_id, phonetic_match,
      "fuzzy_matcher": DataFrame with record1_id, record2_id, fuzzy_score
      # etc.
    }

    config["matching_techniques_config"] must include:
      - exact_match_weight
      - normalized_match_weight
      - phonetic_similarity.weight
      - string_similarity.weight
      - graph_based.weight (if used)
      - ml_based_dedupe.weight (if used)
    config["aggregation_method"] is either "weighted_average" or "max_wins"
    """
    method = config.get("aggregation_method", "weighted_average")
    tech_cfg = config.get("matching_techniques_config", {})

    # Build a dict: (id1, id2) → { exact_score, normalized_score, phonetic_score, similarity_score, graph_score, ml_score }
    pair_scores: Dict[tuple, Dict[str, float]] = {}

    for matcher_name, df in match_results.items():
        for _, row in df.iterrows():
            id1, id2 = row["record1_id"], row["record2_id"]
            key = tuple(sorted((id1, id2)))
            if key not in pair_scores:
                pair_scores[key] = {
                    "exact_score": 0.0,
                    "normalized_score": 0.0,
                    "phonetic_score": 0.0,
                    "similarity_score": 0.0,
                    "graph_score": 0.0,
                    "ml_score": 0.0,
                }

            if matcher_name == "fuzzy_matcher":
                pair_scores[key]["similarity_score"] = float(row.get("fuzzy_score", 0.0))
            elif matcher_name == "exact_matcher":
                pair_scores[key]["exact_score"] = 1.0
            elif matcher_name == "normalized_matcher":
                pair_scores[key]["normalized_score"] = 1.0
            elif matcher_name == "phonetic_matcher":
                pair_scores[key]["phonetic_score"] = 1.0
            elif matcher_name == "graph_matcher":
                pair_scores[key]["graph_score"] = float(row.get("graph_score", 0.0))
            elif matcher_name == "ml_spark_matcher":
                pair_scores[key]["ml_score"] = float(row.get("ml_score", 0.0))
            # Add other matcher mappings as needed

    rows = []
    for (id1, id2), scores in pair_scores.items():
        scores_present = []
        weights_present = []

        # Exact
        es = scores["exact_score"]
        if es > 0:
            scores_present.append(es)
            weights_present.append(tech_cfg.get("exact_match_weight", 1.0))

        # Normalized
        ns = scores["normalized_score"]
        if ns > 0:
            scores_present.append(ns)
            weights_present.append(tech_cfg.get("normalized_match_weight", 0.9))

        # Similarity (fuzzy)
        ss = scores["similarity_score"]
        if ss > 0:
            scores_present.append(ss)
            weights_present.append(tech_cfg.get("string_similarity", {}).get("weight", 0.7))

        # Phonetic
        ps = scores["phonetic_score"]
        if ps > 0:
            scores_present.append(ps)
            weights_present.append(tech_cfg.get("phonetic_similarity", {}).get("weight", 0.6))

        # Graph
        gs = scores["graph_score"]
        if gs > 0 and tech_cfg.get("graph_based", {}).get("enabled", False):
            scores_present.append(gs)
            weights_present.append(tech_cfg.get("graph_based", {}).get("weight", 0.0))

        # ML
        ms = scores["ml_score"]
        if ms > 0 and tech_cfg.get("ml_based_dedupe", {}).get("enabled", False):
            scores_present.append(ms)
            weights_present.append(tech_cfg.get("ml_based_dedupe", {}).get("weight", 0.0))

        if not scores_present:
            final = 0.0
        elif method == "weighted_average":
            # Only use weights corresponding to scores > 0
            weighted_sum = np.dot(scores_present, weights_present)
            total_weight = sum(weights_present)
            final = weighted_sum / total_weight if total_weight else 0.0
        elif method == "max_wins":
            final = float(max(scores_present))
        else:
            raise EnsembleError(f"Unsupported aggregation method: {method}")

        rows.append({
            "record1_id": id1,
            "record2_id": id2,
            "exact_score": scores["exact_score"],
            "normalized_score": scores["normalized_score"],
            "phonetic_score": scores["phonetic_score"],
            "similarity_score": scores["similarity_score"],
            "graph_score": scores["graph_score"],
            "ml_score": scores["ml_score"],
            "final_score": round(final, 4)
        })

    result_df = pd.DataFrame(rows)
    logger.info(f"Ensembled {len(result_df)} pairs using '{method}'.")
    return result_df
