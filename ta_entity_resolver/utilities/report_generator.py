# utilities/report_generator.py

import os
import json
import yaml
import re
import logging
import pandas as pd

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CONTENT = (
    "You are an expert report generator for vendor entity resolution analysis. "
    "Your output should be a clear, concise, and professional executive summary and key insights report."
)

def initialize_llm_client(azure_openai_config: dict) -> AzureChatOpenAI:
    deployment_name = azure_openai_config.get("deployment_name")
    endpoint = azure_openai_config.get("endpoint")
    api_key_env_var = azure_openai_config.get("api_key_env_var")
    api_version = azure_openai_config.get("api_version")
    temperature = azure_openai_config.get("temperature", 0.0)
    max_tokens = azure_openai_config.get("max_tokens", None)

    if not all([deployment_name, endpoint, api_key_env_var, api_version]):
        raise ValueError("One or more essential Azure OpenAI values are missing in config.yaml under 'azure_openai'.")

    api_key = os.getenv(api_key_env_var, api_key_env_var)
    if not api_key:
        raise ValueError(f"Azure OpenAI API key not found in environment variable '{api_key_env_var}'.")

    logger.info("Initializing AzureChatOpenAI client with deployment: %s", deployment_name)
    return AzureChatOpenAI(
        azure_deployment=deployment_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens
    )

def parse_pipeline_log(log_file_path: str) -> dict:
    summary = {
        "total_records": "N/A",
        "record_pairs_evaluated": "N/A",
        "matched_pairs": "N/A",
        "clusters_formed": "N/A",
        "time_taken_seconds": "N/A"
    }
    flags = {
        "low_confidence_clusters": 0,
        "llm_review_required": 0
    }

    if not os.path.exists(log_file_path):
        logger.warning(f"Log file '{log_file_path}' not found. Some summary data will be 'N/A'.")
        return {
            "run_summary": summary,
            "top_clusters": [],
            "flags": flags
        }

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if m := re.search(r'Loaded input .* -> (\d+) rows,', line):
                summary["total_records"] = int(m.group(1))
            if m := re.search(r'Fuzzy matcher found (\d+) matched pairs;', line):
                summary["matched_pairs"] = int(m.group(1))
            if m := re.search(r'Completed (\d+) comparisons\.', line):
                summary["record_pairs_evaluated"] = int(m.group(1))
            if m := re.search(r'Generated ensemble output with (\d+) rows\.', line):
                summary["clusters_formed"] = int(m.group(1))

    summary["time_taken_seconds"] = 0.0  # Optional enhancement: extract from log
    return {
        "run_summary": summary,
        "top_clusters": [],
        "flags": flags
    }

def analyze_final_results_csv(csv_file_path: str) -> dict:
    results_summary = {
        "total_matched_pairs_csv": 0,
        "average_final_score": "N/A",
        "score_distribution": {},
        "top_scoring_pairs": [],
        "inferred_top_clusters": []
    }
    top_clusters_details = []

    if not os.path.exists(csv_file_path):
        logger.warning(f"Final results CSV '{csv_file_path}' not found. Skipping CSV analysis.")
        return results_summary

    try:
        df = pd.read_csv(csv_file_path)
        df['is_match'] = df['is_match'].astype(str).str.lower() == 'true'
        matched_df = df[df['is_match'] == True]

        results_summary["total_matched_pairs_csv"] = len(matched_df)

        if not matched_df.empty:
            results_summary["average_final_score"] = matched_df['final_score'].mean()

            for col in ['levenshtein_score', 'jaro_winkler_score', 'token_set_ratio_score', 'exact_score']:
                if col in matched_df.columns:
                    results_summary["score_distribution"][col] = {
                        "average": matched_df[col].mean(),
                        "min": matched_df[col].min(),
                        "max": matched_df[col].max()
                    }

            top_pairs = matched_df.nlargest(5, 'final_score')[['record1_id', 'record2_id', 'final_score']].to_dict(orient='records')
            results_summary["top_scoring_pairs"] = top_pairs

            all_ids = pd.concat([matched_df['record1_id'], matched_df['record2_id']]).dropna().astype(int)
            top_n_ids = all_ids.value_counts().nlargest(5).index.tolist()

            for record_id in top_n_ids:
                related = matched_df[(matched_df['record1_id'] == record_id) | (matched_df['record2_id'] == record_id)]
                members = pd.concat([related['record1_id'], related['record2_id']]).dropna().unique().astype(int)
                top_clusters_details.append({
                    "cluster_representative_id": int(record_id),
                    "members_sample": [str(m) for m in members[:4]],
                    "avg_match_score": round(related['final_score'].mean(), 4),
                    "num_matched_pairs_in_cluster": len(related)
                })

            results_summary["inferred_top_clusters"] = top_clusters_details

    except Exception as e:
        logger.error(f"Error analyzing final_results.csv: {e}", exc_info=True)

    return results_summary

def get_llm_summary_report(llm_client: AzureChatOpenAI, analysis_data: dict) -> str:
    prompt_content = f"""
    Generate a concise executive summary and key insights report for a vendor entity resolution run.
    Focus on the overall performance, significant findings like top clusters, match score statistics, and any flags that require attention.

    Start with a very brief (3-4 sentences) executive summary for non-technical readers.

    Here is the analysis data in JSON format:
    {json.dumps(analysis_data, indent=2)}

    Please structure your report clearly, including:
    - Overall run statistics.
    - Key score statistics (interpreted, not raw).
    - Highlights of top clusters (3-4 records).
    - Flags with their counts.
    - A short "Recommendations" section.

    Format as clear text only. No code blocks.
    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_CONTENT),
        HumanMessage(content=prompt_content),
    ]
    try:
        response = llm_client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error("LLM generation error: %s", e, exc_info=True)
        return f"Error generating summary report: {e}"

def run_report_generation(config: dict) -> str:
    azure_openai_cfg = config.get("azure_openai", {})
    if not azure_openai_cfg:
        raise ValueError("Missing 'azure_openai' config section.")

    pipeline_log_path = config.get("output", {}).get("pipeline_log_path", "output/pipeline.log")
    final_results_path = config.get("output", {}).get("final_results_path", "output/final_results.csv")
    report_output_path = config.get("output", {}).get("llm_report_path", "output/llm_summary_report.txt")

    logger.info(f"Parsing pipeline log: {pipeline_log_path}")
    log_summary = parse_pipeline_log(pipeline_log_path)

    logger.info(f"Analyzing final_results.csv: {final_results_path}")
    csv_summary = analyze_final_results_csv(final_results_path)

    report_input = {
        "run_summary": log_summary["run_summary"],
        "flags": log_summary["flags"],
        "final_results_summary": csv_summary,
        "top_clusters": csv_summary["inferred_top_clusters"]
    }

    logger.info("Calling LLM to generate summary report")
    llm_client = initialize_llm_client(azure_openai_cfg)
    summary_text = get_llm_summary_report(llm_client, report_input)

    # Handle local or DBFS write
    try:
        os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        logger.info(f"LLM summary written to {report_output_path}")
    except Exception as e:
        logger.warning(f"Could not write report to local/DBFS path '{report_output_path}': {e}")

    return summary_text
