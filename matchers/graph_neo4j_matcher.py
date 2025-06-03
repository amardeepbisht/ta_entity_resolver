import sys
import re
from typing import List, Dict
from pyspark.sql.functions import col, trim
from neo4j import GraphDatabase
import pandas as pd

def to_pascal_case(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]', ' ', s).strip()
    return ''.join(word.capitalize() for word in s.split())

class DatabricksGraphLoader:
    def __init__(self, config: dict):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.get("neo4j_uri"), 
            auth=(config.get("neo4j_user"), config.get("neo4j_password"))
        )
        self.labels = []
        self.relationships = []

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session(database=self.config.get("neo4j_database")) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def setup_neo4j_config(self):
        spark.conf.set("neo4j.url", self.config.get("neo4j_uri"))
        spark.conf.set("neo4j.authentication.basic.username", self.config.get("neo4j_user"))
        spark.conf.set("neo4j.authentication.basic.password", self.config.get("neo4j_password"))
        spark.conf.set("neo4j.database", self.config.get("neo4j_database"))

    def load_graph_spark(self, df, id_col: str, entity_cols: List[str]):
        record_df = df.select(id_col).withColumnRenamed(id_col, "id")
        record_df.write.format("org.neo4j.spark.DataSource") \
            .mode("Overwrite") \
            .option("labels", ":Record") \
            .option("node.keys", "id") \
            .save()

        for entity_col in entity_cols:
            label = to_pascal_case(entity_col)
            rel = f"HAS_{label.upper()}"
            self.labels.append(label)
            self.relationships.append(rel)

            clean_df = df.select(id_col, entity_col) \
                .filter((col(entity_col).isNotNull()) & (trim(col(entity_col)) != "")) \
                .withColumnRenamed(id_col, "id") \
                .withColumnRenamed(entity_col, "value")

            clean_df.select("value").dropDuplicates() \
                .write.format("org.neo4j.spark.DataSource") \
                .mode("Append") \
                .option("labels", f":{label}") \
                .option("node.keys", "value") \
                .save()

            clean_df.write.format("org.neo4j.spark.DataSource") \
                .mode("Append") \
                .option("relationship", rel) \
                .option("relationship.save.strategy", "keys") \
                .option("relationship.source.labels", ":Record") \
                .option("relationship.source.node.keys", "id:id") \
                .option("relationship.target.labels", f":{label}") \
                .option("relationship.target.node.keys", "value:value") \
                .save()

    def create_similarity_edges(self, match_config: Dict):
        with self.driver.session(database=self.config.get("neo4j_database")) as session:
            for col, config in match_config.items():
                label = to_pascal_case(col)
                rel_type = f"HAS_{label.upper()}"
                if config["type"] == "exact":
                    query = f"""
                        MATCH (a:Record)-[:{rel_type}]->(x:{label}),
                              (b:Record)-[:{rel_type}]->(x)
                        WHERE a.id < b.id
                        MERGE (a)-[r:SIMILAR_TO]->(b)
                        ON CREATE SET r.score = 1.0, r.reasons = ['{label}']
                        ON MATCH SET r.score = r.score + 1.0, r.reasons = r.reasons + '{label}'
                    """
                    session.run(query)
                else:
                    query = f"""
                        MATCH (a:Record)-[:{rel_type}]->(t1:{label}),
                              (b:Record)-[:{rel_type}]->(t2:{label})
                        WHERE a.id < b.id 
                        WITH a, b, apoc.text.levenshteinSimilarity(toLower(t1.value), toLower(t2.value)) AS sim_score
                        WHERE sim_score >= $threshold
                        MERGE (a)-[r:SIMILAR_TO]->(b)
                        ON CREATE SET r.score = sim_score, r.reasons = ['{label}']
                        ON MATCH SET r.score = r.score + sim_score, r.reasons = r.reasons + '{label}'
                    """
                    session.run(query, {"threshold": config["threshold"]})

    def filter_similarity_edges(self, score_threshold: float, priority_columns: List[str]):
        with self.driver.session(database=self.config.get("neo4j_database")) as session:
            priority_labels = [to_pascal_case(col) for col in priority_columns]
            cypher_list = "[" + ", ".join(f'"{label}"' for label in priority_labels) + "]"
            where_clause = f"""
                WHERE r.score < $threshold AND
                      NONE(p IN {cypher_list} WHERE p IN r.reasons)
            """ if priority_columns else "WHERE r.score < $threshold"

            session.run(f"""
                MATCH (a)-[r:SIMILAR_TO]->(b)
                {where_clause}
                DELETE r
            """, {"threshold": score_threshold})

    def compute_weighted_degree_centrality(self):
        with self.driver.session(database=self.config.get("neo4j_database")) as session:
            session.run("CALL gds.graph.drop('sub_graph_d', false) YIELD graphName")
            session.run("""
                CALL gds.graph.project(
                    'sub_graph_d',
                    'Record',
                    {
                        SIMILAR_TO: {
                            type: 'SIMILAR_TO',
                            orientation: 'UNDIRECTED',
                            properties: {
                                score: {
                                    property: 'score'
                                }
                            }
                        }
                    }
                )
            """)
            session.run("""
                CALL gds.degree.write('sub_graph_d', {
                    writeProperty: 'degree_centrality',
                    relationshipWeightProperty: 'score'
                })
            """)

    def run_clustering(self):
        with self.driver.session(database=self.config.get("neo4j_database")) as session:
            session.run("CALL gds.graph.drop('sub_graph', false) YIELD graphName")
            session.run("""
                CALL gds.graph.project(
                    'sub_graph',
                    'Record',
                    'SIMILAR_TO'
                )
            """)
            session.run("""
                CALL gds.wcc.write(
                    'sub_graph',
                    {
                        writeProperty: 'cluster_id'
                    }
                )
            """)
        self.compute_weighted_degree_centrality()

    def get_clustering_results(self) -> pd.DataFrame:
        with self.driver.session(database=self.config.get("neo4j_database")) as session:
            match_parts = ["MATCH (r:Record)"]
            return_parts = ["r.id AS record_id", "r.cluster_id AS cluster_id", "r.degree_centrality AS centrality"]
            for label in self.labels:
                alias = label.lower()
                rel_type = f"HAS_{alias.upper()}"
                match_parts.append(f"OPTIONAL MATCH (r)-[:{rel_type}]->({alias}:{label})")
                return_parts.append(f"{alias}.value AS {alias}_value")
            query = "\n".join(match_parts) + "\nRETURN " + ", ".join(return_parts) + "\nORDER BY r.cluster_id, centrality DESC"
            result = session.run(query)
            return pd.DataFrame([dict(r) for r in result])

def main(config: dict):
    loader = DatabricksGraphLoader(config)
    loader.setup_neo4j_config()
    loader.clear_database()

    df = spark.read.table(config.get("table_name"))
    id_col = config.get("id_column")
    entity_cols = config.get("entity_columns")
    priority_cols = config.get("priority_columns", [])

    loader.load_graph_spark(df, id_col, entity_cols)

    match_types = config.get("match_types")
    thresholds = config.get("fuzzy_thresholds", [])

    match_config = {}
    for col, mtype in zip(entity_cols, match_types):
        threshold = float(thresholds.pop(0)) if mtype == 'fuzzy' and thresholds else 1.0
        match_config[col] = {"type": mtype, "threshold": threshold}

    loader.create_similarity_edges(match_config)

    score_threshold = sum(1.0 if v["type"] == "exact" else v["threshold"] for v in match_config.values())
    loader.filter_similarity_edges(score_threshold, priority_cols)

    loader.run_clustering()
    result_df = loader.get_clustering_results()
    display(result_df)


if __name__ == "__main__": 
    with open("config.json") as f:
        config = json.load(f)
    main(config)