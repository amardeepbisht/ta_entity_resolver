# matcher_engine\matcher_runner.py

import sys
import os
import importlib
import logging
from resolver_config.config_loader import load_config
from preprocessor.preprocessor import preprocess_data
from data_loader.load_data import load_input_data

logger = logging.getLogger(__name__)


def matcher_runner(df, config, engine="pyspark"):
    """
    Executes matchers based on:
      1) Which techniques fields request (fields_to_match[].techniques.<tech>=True)
      2) Whether top-level config["matchers"]["use_<tech>"] is True

    Dynamically locates and calls run_<tech>_matcher functions in matchers/<tech>_matcher.py.
    Logs warnings if a field requests a technique whose matcher is missing or disabled.
    Returns a dict of DataFrames keyed by "<tech>_matcher".
    """
    results = {}
    matchers_conf = config.get("matchers", {})
    fields = config.get("fields_to_match", [])

    # 1. Gather all techniques requested by any field
    needed_techniques = set()
    for f in fields:
        techniques = f.get("techniques", {})
        needed_techniques.update({tech for tech, enabled in techniques.items() if enabled})

    # 2. For each requested technique, attempt to run it if enabled
    for tech in sorted(needed_techniques):
        top_flag = f"use_{tech}"
        if not matchers_conf.get(top_flag, False):
            logger.warning(
                "Field requests '%s' matching, but top-level '%s' is false. Skipping %s matcher.",
                tech, top_flag, tech
            )
            continue

        module_name = f"matchers.{tech}_matcher"
        func_name = f"run_{tech}_matcher"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            logger.warning(
                "No module '%s' found for technique '%s'. Skipping.", module_name, tech
            )
            continue

        runner_fn = getattr(module, func_name, None)
        if not callable(runner_fn):
            logger.warning(
                "Module '%s' does not define function '%s'. Skipping.", module_name, func_name
            )
            continue

        logger.info("Running %s Matcher...", tech.capitalize())
        try:
            df_matched = runner_fn(df, config)
        except Exception as e:
            logger.error("Error while running %s matcher: %s", tech, e)
            continue

        results[f"{tech}_matcher"] = df_matched

    # 3. Handle ML Spark matcher (controlled purely by top-level flag)
    if matchers_conf.get("use_ml_spark_matcher", False):
        if engine != "pyspark":
            raise ValueError("ML Spark Matcher requires engine='pyspark'")
        try:
            from matchers.ml_spark_matcher import run_ml_spark_matcher
            logger.info("Running ML Spark Matcher...")
            df_spark = run_ml_spark_matcher(df, config)
            results["ml_spark_matcher"] = df_spark
        except Exception as e:
            logger.error("Error while running ml_spark_matcher: %s", e)

    # 4. Handle Graph matcher (controlled purely by top-level flag)
    if matchers_conf.get("use_graph", False):
        try:
            from matchers.graph_matcher import run_graph_matcher
            logger.info("Running Graph Matcher...")
            df_graph = run_graph_matcher(df, config)
            results["graph_matcher"] = df_graph
        except ModuleNotFoundError:
            logger.warning("Module 'matchers.graph_matcher' not found; skipping.")
        except Exception as e:
            logger.error("Error while running graph_matcher: %s", e)

    return results
