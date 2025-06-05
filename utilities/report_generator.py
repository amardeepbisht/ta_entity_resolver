# utilities/report_generator.py

import os
import json
import yaml
import re
import logging
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CONTENT = (
    "You are an expert report generator for vendor entity "
    "resolution analysis. Your output should be a clear, concise, "
    "and professional executive summary and key insights report."
)

def initialize_llm_client(azure_openai_config: dict) -> AzureChatOpenAI:
    """Initializes and returns the AzureChatOpenAI LLM client using provided config."""
    deployment_name = azure_openai_config.get("deployment_name")
    endpoint = azure_openai_config.get("endpoint")
    api_key_env_var = azure_openai_config.get("api_key_env_var")
    api_version = azure_openai_config.get("api_version")
    temperature = azure_openai_config.get("temperature", 0.0)
    max_tokens = azure_openai_config.get("max_tokens", None)

    if not all([deployment_name, endpoint, api_key_env_var, api_version]):
        raise ValueError("One or more essential Azure OpenAI values are missing in config.yaml under 'azure_openai'.")

    # If api_key_env_var is truly an environment‐variable name, use os.getenv:
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
    """
    Parses the pipeline log file to extract key entity resolution metrics.
    Returns a dict with run_summary, top_clusters (empty), flags.
    """
    summary = {
        "total_records": "N/A",
        "record_pairs_evaluated": "N/A",
        "matched_pairs": "N/A",
        "clusters_formed": "N/A",
        "time_taken_seconds": "N/A"
    }
    top_clusters = []
    flags = {
        "low_confidence_clusters": 0,
        "llm_review_required": 0
    }

    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file '{log_file_path}' not found.")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'Loaded input .* -> (\d+) rows,', line)
            if m:
                summary["total_records"] = int(m.group(1))

            m = re.search(r'Fuzzy matcher found (\d+) matched pairs;', line)
            if m:
                summary["matched_pairs"] = int(m.group(1))

            m = re.search(r'Completed (\d+) comparisons\.', line)
            if m:
                summary["record_pairs_evaluated"] = int(m.group(1))

            m = re.search(r'Generated ensemble output with (\d+) rows\.', line)
            if m:
                summary["clusters_formed"] = int(m.group(1))

    # As before, time_taken_seconds and flags remain placeholders unless you log them explicitly
    summary["time_taken_seconds"] = 0.0

    return {
        "run_summary": summary,
        "top_clusters": top_clusters,
        "flags": flags
    }


def get_llm_summary_report(llm_client: AzureChatOpenAI, analysis_data: dict) -> str:
    """
    Uses the LLM to generate a human‐readable summary based on `analysis_data`.
    """
    prompt_content = f"""
Generate a concise executive summary and key insights report for a vendor entity resolution run.
Focus on the overall performance, significant findings like top clusters, and any flags that require attention.

Here is the analysis data in JSON format:
{json.dumps(analysis_data, indent=2)}

Please structure your report clearly, including:
- Overall run statistics (total records, time, matched pairs, clusters formed).
- Details of the top clusters, listing a few key members (up to 3-4) for each.
- Any flagged items, specifically mentioning the counts for low confidence clusters and LLM review required clusters.
- A brief "Recommendations" section based on the flags and overall insights.

Format the output as a clear, easy-to-read text summary. Do not include any JSON fences or code blocks in your final output.
    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_CONTENT),
        HumanMessage(content=prompt_content),
    ]
    try:
        response = llm_client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error("Error calling LLM for summary report: %s", e, exc_info=True)
        return f"Error generating summary report: {e}"


def run_report_generation(config: dict) -> str:
    """
    Orchestrates parsing the pipeline log and calling the LLM to generate the report.
    Returns the plain‐text summary.
    """

    # 1) Read Azure OpenAI config
    azure_openai_cfg = config.get("azure_openai", {})
    if not azure_openai_cfg:
        raise ValueError("Missing 'azure_openai' section in config.")

    # 2) Read pipeline log path from config
    pipeline_log_path = config.get("output", {}).get("pipeline_log_path", "output/pipeline.log")
    if not os.path.exists(pipeline_log_path):
        raise FileNotFoundError(f"Pipeline log not found at '{pipeline_log_path}'.")

    logger.info("Parsing pipeline log: %s", pipeline_log_path)
    analysis_data = parse_pipeline_log(pipeline_log_path)

    # 3) Initialize LLM client
    llm_client = initialize_llm_client(azure_openai_cfg)

    # 4) Generate the summary text
    logger.info("Calling LLM to generate summary report")
    summary_text = get_llm_summary_report(llm_client, analysis_data)
    logger.info("Received summary report from LLM")

    # 5) Write summary to disk
    report_output_path = config.get("output", {}).get("llm_report_path", "output/llm_summary_report.txt")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info("Saved LLM summary report to: %s", report_output_path)

    return summary_text
