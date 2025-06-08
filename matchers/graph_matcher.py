# matchers/graph_matcher.py
import os
import re
import logging
import pandas as pd
from typing import List, Dict
from neo4j import GraphDatabase
from pandas import DataFrame
from dotenv import load_dotenv
load_dotenv()

# Create a module-level logger
logger = logging.getLogger(__name__)


def to_pascal_case(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s).strip()
    return ''.join(word.capitalize() for word in s.split())


import os
import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class LocalGraphLoader:
    def __init__(self, config: dict):
        uri = config.get("neo4j_uri")
        user = config.get("neo4j_user")
        database = config.get("neo4j_database")

        logger.info("Initializing LocalGraphLoader with Neo4j URI: %s", uri)

        # Get environment variable name from config
        env_var_name = config.get("neo4j_password_env_var")
        if not env_var_name:
            raise ValueError("Missing 'neo4j_password_env_var' in graph_matcher config.")

        pw = os.getenv(env_var_name)
        if not pw:
            raise ValueError(f"Environment variable '{env_var_name}' is not set or is empty.")

        # 3. Now create the driver using (user, password)
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, pw)
            )
            self.database = database
            self.labels = []
            self.relationships = []
            logger.info("Neo4j driver successfully created; database=%s", self.database)
        except Exception as e:
            logger.error("Failed to create Neo4j driver: %s", e, exc_info=True)
            raise


    def close(self):
        logger.info("Closing Neo4j driver connection")
        try:
            self.driver.close()
        except Exception as e:
            logger.warning("Error while closing Neo4j driver: %s", e)

    def clear_database(self):
        logger.info("Clearing Neo4j database: %s", self.database)
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error("Error clearing database: %s", e, exc_info=True)
            raise

    def load_graph_batch(self, df: DataFrame, id_col: str, entity_cols: List[str], batch_size: int = 1000):
        total = len(df)
        logger.info("Loading graph in batches: total_records=%d, batch_size=%d", total, batch_size)
        try:
            with self.driver.session(database=self.database) as session:
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch = df.iloc[start:end]
                    logger.info("Processing batch %d to %d", start, end - 1)

                    for idx, row in batch.iterrows():
                        rid = str(row[id_col])
                        session.run("MERGE (r:Record {id: $id})", {"id": rid})

                        for col in entity_cols:
                            val = row[col]
                            if pd.isna(val) or str(val).strip() == "":
                                continue
                            label = to_pascal_case(col)
                            rel = f"HAS_{label.upper()}"
                            self.labels.append(label)
                            self.relationships.append(rel)
                            session.run(
                                f"""
                                MERGE (e:{label} {{value: $value}})
                                WITH e
                                MATCH (r:Record {{id: $id}})
                                MERGE (r)-[:{rel}]->(e)
                                """,
                                {"value": str(val).strip(), "id": rid}
                            )
                    logger.info("Completed batch %d to %d", start, end - 1)
            logger.info("Finished loading all %d records into graph", total)
        except Exception as e:
            logger.error("Error during load_graph_batch: %s", e, exc_info=True)
            raise

    def create_similarity_edges(self, match_config: Dict[str, Dict], factor: int):
        logger.info("Creating similarity edges with factor=%d", factor)
        try:
            with self.driver.session(database=self.database) as session:
                for col, config in match_config.items():
                    label = to_pascal_case(col)
                    rel_type = f"HAS_{label.upper()}"
                    mtype = config["type"]
                    threshold = config.get("threshold", None)
                    logger.info("Creating edges for column='%s', type='%s', threshold=%s", col, mtype, threshold)

                    if mtype == "exact":
                        query = f"""
                            MATCH (a:Record)-[:{rel_type}]->(x:{label}),
                                  (b:Record)-[:{rel_type}]->(x)
                            WHERE a.id < b.id
                            MERGE (a)-[r:SIMILAR_TO]->(b)
                            ON CREATE SET r.score = 1.0/{factor}, r.reasons = [$col]
                            ON MATCH SET r.score = r.score + 1.0/{factor}, r.reasons = r.reasons + $col
                        """
                        session.run(query, {"col": label})
                        logger.debug("Exact similarity query executed for label=%s", label)
                    else:
                        query = f"""
                            MATCH (a:Record)-[:{rel_type}]->(t1:{label}),
                                  (b:Record)-[:{rel_type}]->(t2:{label})
                            WHERE a.id < b.id 
                            WITH a, b, t1, t2,
                                CASE 
                                    WHEN toLower(t1.value) = toLower(t2.value) THEN 1.0 
                                    ELSE apoc.text.levenshteinSimilarity(toLower(t1.value), toLower(t2.value)) 
                                END AS sim_score
                            WHERE sim_score >= $threshold
                            MERGE (a)-[r:SIMILAR_TO]->(b)
                            ON CREATE SET r.score = sim_score / {factor}, r.reasons = [$col]
                            ON MATCH SET r.score = r.score + sim_score / {factor}, r.reasons = r.reasons + $col
                        """
                        session.run(query, {"threshold": threshold, "col": label})
                        logger.debug("Fuzzy similarity query executed for label=%s with threshold=%s", label, threshold)
            logger.info("Finished creating similarity edges for all columns")
        except Exception as e:
            logger.error("Error during create_similarity_edges: %s", e, exc_info=True)
            raise

    def get_similarity_edges(self) -> pd.DataFrame:
        logger.info("Retrieving similarity edges from Neo4j")
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (a:Record)-[r:SIMILAR_TO]->(b:Record)
                    WHERE a.id < b.id
                    RETURN a.id AS record1_id, b.id AS record2_id, r.score AS score, r.reasons AS reasons
                    ORDER BY score DESC
                """)
                records = [dict(row) for row in result]
                df = pd.DataFrame(records)
                logger.info("Retrieved %d similarity edges", len(df))
                return df
        except Exception as e:
            logger.error("Error retrieving similarity edges: %s", e, exc_info=True)
            raise


def run_graph_matcher(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    graph_cfg = config.get("matching_techniques_config", {}).get("graph_matcher", {})
    if not graph_cfg or not config.get("matchers", {}).get("use_graph", False):
        raise ValueError("Graph matcher is not enabled or missing configuration.")

    logger.info("Running Graph Matcher")

    # The preprocessor has already inserted 'record_id'
    record_id_col = config.get("record_id_column", "record_id")

    entity_cols = graph_cfg["entity_columns"]
    match_types = graph_cfg["match_types"]
    thresholds = graph_cfg.get("fuzzy_thresholds", []).copy()

    # Construct match_config
    match_config = {}
    for col, mtype in zip(entity_cols, match_types):
        threshold = float(thresholds.pop(0)) if mtype == 'fuzzy' and thresholds else 1.0
        match_config[col] = {"type": mtype, "threshold": threshold}
    logger.debug("Match configuration: %s", match_config)

    loader = LocalGraphLoader(graph_cfg)
    factor = len(entity_cols)

    try:
        loader.clear_database()
        loader.load_graph_batch(df, record_id_col, entity_cols)
        loader.create_similarity_edges(match_config, factor)

        matched_df = loader.get_similarity_edges()

        logger.debug("Graph matcher config: %s", graph_cfg)
        global_thresh = float(graph_cfg.get("global_threshold", 0.0))
        logger.info("Applying graph global_threshold = %.2f", global_thresh)

        matched_df = matched_df.rename(columns={"score": "graph_score"})
        matched_df["score_type"] = "graph_score"
        matched_df["source"] = "graph_matcher"

        # --- Remove 'reasons' and normalize columns ---
        if "reasons" in matched_df.columns:
            matched_df = matched_df.drop(columns=["reasons"])

        #cast the record IDs to int
        matched_df["record1_id"] = matched_df["record1_id"].astype(int)
        matched_df["record2_id"] = matched_df["record2_id"].astype(int)

   
        #added global threshold filtering
        global_thresh = graph_cfg.get("global_threshold", 0.0)
        if global_thresh > 0.0:
            logger.info("Filtering graph matches with global_threshold=%.2f", global_thresh)
            matched_df = matched_df[matched_df["graph_score"] >= global_thresh]
            logger.info("After filtering: %d edges remain", len(matched_df))
        matched_df["score_type"] = "graph_score"
        matched_df["source"] = "graph_matcher"


        logger.info("Graph Matcher produced %d matched pairs", len(matched_df))

        out_base = config["output"].get("graph_matcher_path", "output/graph_matcher.csv")
        logger.info("Writing Graph Matcher results to CSV at: %s", out_base)
        out_dir = os.path.dirname(out_base)
        os.makedirs(out_dir, exist_ok=True)
        matched_df.to_csv(out_base, index=False)
        logger.info("Saved Graph Matcher output successfully to %s", out_base)


        return matched_df
    finally:
        loader.close()
