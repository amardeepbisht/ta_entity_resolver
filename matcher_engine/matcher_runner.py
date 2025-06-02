import logging
from config.config_loader import load_config
from matchers.ml_spark_matcher import run_ml_spark_matcher
from preprocessor.preprocessor import preprocess_data
# from matchers.exact_matcher import run_exact_matcher  # Future
# from matchers.graph_matcher import run_graph_matcher  # Future
# from matchers.zingg_matcher import run_zingg_matcher  # Future


def matcher_runner(df, config, engine="pyspark"):
    """
    Executes selected matchers based on config.
    Returns a dictionary of match results keyed by matcher name.
    """
    logger = logging.getLogger(__name__)
    results = {}

    matchers_config = config.get("matchers", {})

    if matchers_config.get("use_ml_spark_matcher", False):
        logger.info("Running ML Spark Matcher...")

    if engine != "pyspark":
       raise ValueError("ML Spark Matcher requires engine to be 'pyspark'. Please update your config.")
    
    result_df = run_ml_spark_matcher(df, config)
    results["ml_spark_matcher"] = result_df

    # Future extensions (placeholders):
    # if matchers_config.get("use_exact", False):
    #     logger.info("Running Exact Matcher...")
    #     results["exact"] = run_exact_matcher(df, config)

    # if matchers_config.get("use_graph", False):
    #     logger.info("Running Graph Matcher...")
    #     results["graph"] = run_graph_matcher(df, config)

    # if matchers_config.get("use_ml_zingg", False):
    #     logger.info("Running Zingg Matcher...")
    #     results["ml_zingg"] = run_zingg_matcher(df, config)

    return results


if __name__ == "__main__":
    import sys
    from data_loader.load_data import load_input_data

    logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_config("config/config.yaml")
    df = load_input_data(config["input"], engine=config.get("engine", "pandas"))
    df = preprocess_data(df, config)  

    print(df.head())
    print(config)
   
    matcher_outputs = matcher_runner(df, config)
    for name, result_df in matcher_outputs.items():
        print(f"\nMatcher: {name}, Rows: {result_df.count() if hasattr(result_df, 'count') else len(result_df)}")

