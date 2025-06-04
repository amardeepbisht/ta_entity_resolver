# postprocessing/ensembler.py

import os
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def ensemble_scores(match_results: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Combine scores from all matchers (exact, fuzzy, etc.) into final_score/is_match.
    Special rule: if exact_score == 1.0, final_score = 1.0 and skip other matchers.
    """

    # 0. If not using advanced ensemble, run default simple logic
    use_advanced = config.get("use_advanced_ensemble", False)
    if not use_advanced:
        return _simple_ensemble(match_results, config)

    # -------------------------------------
    # BEGIN: Advanced‐ensemble logic
    # -------------------------------------

    # 1. Read shared config keys
    method = config.get("aggregation_method", "weighted_average")
    ensemble_thresh = config.get("ensemble_threshold", 0.75)
    matcher_weights = config.get("weights", {})

    # 2. Read fuzzy sub‐weights 
    raw_fuzzy_sub = config.get("fuzzy_sub_weights", {})
    fuzzy_weight = matcher_weights.get("fuzzy_matcher", 0.0)

    if raw_fuzzy_sub:
        sum_sub = sum(raw_fuzzy_sub.values())
        if abs(sum_sub - fuzzy_weight) > 1e-6:
            logger.warning(
                "fuzzy_sub_weights sum (%.4f) != fuzzy_matcher weight (%.4f); falling back to equal split.",
                sum_sub, fuzzy_weight
            )
            raw_fuzzy_sub = {}
        else:
            factor = fuzzy_weight / sum_sub
            for k in raw_fuzzy_sub:
                raw_fuzzy_sub[k] *= factor

    # 3. Read majority‐vote param
    required_votes = config.get("required_votes", 1)

    # 4. Determine enabled fuzzy algorithms
    string_conf = config.get("matching_techniques_config", {}).get("string_similarity", {})
    algo_flags = string_conf.get("algorithms", {})
    enabled_algos = [name for name, enabled in algo_flags.items() if enabled]
    num_fuzzy_algos = len(enabled_algos)

    # 5. Build per‐fuzzy‐algo weights
    per_fuzzy_weights: Dict[str, float] = {}
    if raw_fuzzy_sub:
        per_fuzzy_weights = raw_fuzzy_sub
    else:
        if num_fuzzy_algos > 0:
            each = fuzzy_weight / num_fuzzy_algos
            for a in enabled_algos:
                per_fuzzy_weights[a] = each
        else:
            per_fuzzy_weights = {a: 0.0 for a in enabled_algos}

    logger.info("Advanced ensemble mode: %s", method)
    logger.info("Ensemble threshold: %.4f", ensemble_thresh)
    logger.info("Matcher weights: %s", matcher_weights)
    logger.info("Per-fuzzy-algo weights: %s", per_fuzzy_weights)
    if method == "majority_vote":
        logger.info("Required votes for majority_vote: %d", required_votes)

    # 6. Collect all per‐pair scores into a dict
    pair_scores: Dict[tuple, Dict[str, float]] = {}

    # 6a. Process fuzzy_matcher DataFrame
    if "fuzzy_matcher" in match_results:
        fuzzy_df = match_results["fuzzy_matcher"]
        fuzzy_score_cols = [c for c in fuzzy_df.columns if c.endswith("_score")]
        logger.info("Fuzzy score columns found: %s", fuzzy_score_cols)

        for _, row in fuzzy_df.iterrows():
            r1, r2 = row["record1_id"], row["record2_id"]
            key = (min(r1, r2), max(r1, r2))
            pair_scores.setdefault(key, {})

            for score_col in fuzzy_score_cols:
                score_raw = row.get(score_col, None)
                if pd.isna(score_raw):
                    continue
                score_val = float(score_raw)
                existing = pair_scores[key].get(score_col)
                pair_scores[key][score_col] = max(existing, score_val) if existing is not None else score_val

    # 6b. Process other matchers (exact, spark, graph, etc.)
    for matcher_name, df in match_results.items():
        if matcher_name == "fuzzy_matcher":
            continue

        score_col = next((c for c in df.columns if c.endswith("_score")), None)
        if not score_col:
            logger.warning("No '*_score' column in DataFrame for matcher '%s'; skipping.", matcher_name)
            continue

        for _, row in df.iterrows():
            r1, r2 = row["record1_id"], row["record2_id"]
            key = (min(r1, r2), max(r1, r2))
            pair_scores.setdefault(key, {})

            score_raw = row.get(score_col, None)
            if pd.isna(score_raw):
                continue
            score_val = float(score_raw)
            existing = pair_scores[key].get(score_col)
            pair_scores[key][score_col] = max(existing, score_val) if existing is not None else score_val

    # 7. Build final rows according to aggregation_method
    final_rows = []
    for (r1, r2), scores_dict in pair_scores.items():
        # 7.1 Exact-first override
        exact_val = scores_dict.get("exact_score")
        if exact_val == 1.0:
            row_out = {
                "record1_id": r1,
                "record2_id": r2,
                "final_score": 1.0,
                "is_match": True
            }
            for col_key, val in scores_dict.items():
                row_out[col_key] = round(val, 4)
            final_rows.append(row_out)
            continue

        # 7.2 Otherwise, aggregate remaining scores
        final_score = 0.0
        is_match = False

        if method == "weighted_average":
            weighted_sum = 0.0
            total_weight = 0.0

            # 7.2a Add fuzzy sub‐algorithm contributions
            for algo, weight in per_fuzzy_weights.items():
                col_key = f"{algo}_score"
                val = scores_dict.get(col_key)
                if val is not None:
                    weighted_sum += val * weight
                    total_weight += weight

            # 7.2b Add other matcher contributions (excluding exact)
            for m_name, w in matcher_weights.items():
                if m_name in ("fuzzy_matcher", "exact_matcher"):
                    continue
                col_key = f"{m_name}_score"
                val = scores_dict.get(col_key)
                if val is not None:
                    weighted_sum += val * w
                    total_weight += w

            final_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
            is_match = final_score >= ensemble_thresh

        elif method == "max_score":
            all_scores = [
                v for k, v in scores_dict.items()
                if k.endswith("_score") and k != "exact_score"
            ]
            if all_scores:
                final_score = max(all_scores)
                is_match = final_score >= ensemble_thresh
            else:
                final_score = 0.0
                is_match = False

        elif method == "min_score":
            all_scores = [
                v for k, v in scores_dict.items()
                if k.endswith("_score") and k != "exact_score"
            ]
            if all_scores:
                final_score = min(all_scores)
                is_match = final_score >= ensemble_thresh
            else:
                final_score = 0.0
                is_match = False

        elif method == "majority_vote":
            votes = 0
            total_matchers = 0

            # 7.2c Fuzzy as one "matcher"
            fuzzy_vals = [
                scores_dict[f"{algo}_score"]
                for algo in enabled_algos
                if scores_dict.get(f"{algo}_score") is not None
            ]
            if fuzzy_vals:
                avg_fuzzy = sum(fuzzy_vals) / len(fuzzy_vals)
                total_matchers += 1
                if avg_fuzzy >= ensemble_thresh:
                    votes += 1

            # 7.2d Other matchers individually (excluding exact)
            for m_name in matcher_weights:
                if m_name == "fuzzy_matcher":
                    continue
                col_key = f"{m_name}_score"
                val = scores_dict.get(col_key)
                if val is not None:
                    total_matchers += 1
                    if val >= ensemble_thresh:
                        votes += 1

            final_score = float(votes)
            is_match = votes >= config.get("required_votes", 1)

        else:
            raise ValueError(f"Unknown aggregation_method: {method}")

        row_out = {
            "record1_id": r1,
            "record2_id": r2,
            "final_score": round(final_score, 4),
            "is_match": is_match
        }
        for col_key, val in scores_dict.items():
            row_out[col_key] = round(val, 4)

        final_rows.append(row_out)

    logger.info("Collected %d unique record pairs from matcher outputs.", len(final_rows))

    final_df = pd.DataFrame(final_rows)

    try:
        out_base = config.get("output", {}).get("final_results_path", "output/final_results.csv")
        out_dir = os.path.dirname(out_base)
        os.makedirs(out_dir, exist_ok=True)
        final_df.to_csv(out_base, index=False)
        logger.info("Saved final ensembled pairs to %s (%d rows)", out_base, len(final_rows))
    except Exception as e:
        logger.warning("Could not save final results CSV: %s", e)

    return final_df


# -----------------------------------------
# Default simple ensemble logic (pre-advanced)
# -----------------------------------------
def _simple_ensemble(match_results: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Original logic (pre-advanced). 
    Uses aggregation_method, ensemble_threshold, weights exactly as before,
    with an added exact-first override.
    """
    method = config.get("aggregation_method", "weighted_average")
    ensemble_thresh = config.get("ensemble_threshold", 0.75)
    matcher_weights = config.get("weights", {})

    string_conf = config.get("matching_techniques_config", {}).get("string_similarity", {})
    algo_flags = string_conf.get("algorithms", {})
    enabled_algos = [name for name, enabled in algo_flags.items() if enabled]
    fuzzy_weight = matcher_weights.get("fuzzy_matcher", 0.0)
    per_fuzzy_weight = (fuzzy_weight / len(enabled_algos)) if enabled_algos else 0.0

    logger.info("Running simple ensemble (pre-advanced).")
    logger.info("Aggregation method: %s, Threshold: %.4f", method, ensemble_thresh)
    logger.info("Matcher Weights: %s", matcher_weights)
    logger.info("Enabled fuzzy algorithms: %s, per-algo weight: %.4f", enabled_algos, per_fuzzy_weight)

    pair_scores: Dict[tuple, Dict[str, float]] = {}

    # Collect fuzzy matcher scores
    if "fuzzy_matcher" in match_results:
        fuzzy_df = match_results["fuzzy_matcher"]
        fuzzy_score_cols = [c for c in fuzzy_df.columns if c.endswith("_score")]
        logger.info("Fuzzy score columns found: %s", fuzzy_score_cols)

        for _, row in fuzzy_df.iterrows():
            r1, r2 = row["record1_id"], row["record2_id"]
            key = (min(r1, r2), max(r1, r2))
            pair_scores.setdefault(key, {})

            for score_col in fuzzy_score_cols:
                score_raw = row.get(score_col, None)
                if pd.isna(score_raw):
                    continue
                score_val = float(score_raw)
                existing = pair_scores[key].get(score_col)
                pair_scores[key][score_col] = max(existing, score_val) if existing is not None else score_val

    # Collect other matcher scores
    for matcher_name, df in match_results.items():
        if matcher_name == "fuzzy_matcher":
            continue
        score_col = next((c for c in df.columns if c.endswith("_score")), None)
        if not score_col:
            continue
        for _, row in df.iterrows():
            r1, r2 = row["record1_id"], row["record2_id"]
            key = (min(r1, r2), max(r1, r2))
            pair_scores.setdefault(key, {})

            score_raw = row.get(score_col, None)
            if pd.isna(score_raw):
                continue
            score_val = float(score_raw)
            existing = pair_scores[key].get(score_col)
            pair_scores[key][score_col] = max(existing, score_val) if existing is not None else score_val

    final_rows = []
    for (r1, r2), scores_dict in pair_scores.items():
        # Exact-first override
        exact_val = scores_dict.get("exact_score")
        if exact_val == 1.0:
            row_out = {
                "record1_id": r1,
                "record2_id": r2,
                "final_score": 1.0,
                "is_match": True
            }
            for col_key, val in scores_dict.items():
                row_out[col_key] = round(val, 4)
            final_rows.append(row_out)
            continue

        # Weighted average excluding exact
        weighted_sum = 0.0
        total_weight = 0.0

        # Fuzzy sub-algorithms
        for algo in enabled_algos:
            col_key = f"{algo}_score"
            val = scores_dict.get(col_key)
            if val is not None:
                weighted_sum += val * per_fuzzy_weight
                total_weight += per_fuzzy_weight

        # Other matchers
        for m_name, w in matcher_weights.items():
            if m_name in ("fuzzy_matcher", "exact_matcher"):
                continue
            col_key = f"{m_name}_score"
            val = scores_dict.get(col_key)
            if val is not None:
                weighted_sum += val * w
                total_weight += w

        final_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        is_match = final_score >= ensemble_thresh

        row_out = {
            "record1_id": r1,
            "record2_id": r2,
            "final_score": round(final_score, 4),
            "is_match": is_match
        }
        for col_key, val in scores_dict.items():
            row_out[col_key] = round(val, 4)

        final_rows.append(row_out)

    logger.info("Collected %d unique record pairs from matcher outputs.", len(final_rows))

    final_df = pd.DataFrame(final_rows)

    try:
        out_base = config.get("output", {}).get("final_results_path", "output/final_results.csv")
        out_dir = os.path.dirname(out_base)
        os.makedirs(out_dir, exist_ok=True)
        final_df.to_csv(out_base, index=False)
        logger.info("Saved final ensembled pairs to %s (%d rows)", out_base, len(final_rows))
    except Exception as e:
        logger.warning("Could not save final results CSV: %s", e)

    return final_df
