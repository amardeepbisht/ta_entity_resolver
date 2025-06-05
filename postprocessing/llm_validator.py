import json
import re
import os
import yaml # Import PyYAML
from dotenv import load_dotenv

# Import Azure-specific chat model
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from resolver_config.config_loader import load_config



# --- System Prompt (can also be moved to config.yaml if preferred) ---
SYSTEM_PROMPT_CONTENT = "You are an expert in vendor record matching that outputs JSON exclusively."


def initialize_llm_client(azure_openai_config: dict) -> AzureChatOpenAI:
    """Initializes and returns the AzureChatOpenAI LLM client usipython -m ng provided config."""
    deployment_name = azure_openai_config.get("deployment_name")
    endpoint = azure_openai_config.get("endpoint")
    load_dotenv()  # Loads from .env file
   
    api_key_env_var = azure_openai_config["api_key_env_var"]
    api_key = os.getenv(api_key_env_var)
    #api_key_env_var = os.getenv("AZURE_OPENAI_API_KEY")
  
    api_version = azure_openai_config.get("api_version")
    temperature = azure_openai_config.get("temperature", 0.0)
    max_tokens = azure_openai_config.get("max_tokens", None)

    # Validate essential configurations
    if not all([deployment_name, endpoint, api_key_env_var, api_version]):
        raise ValueError("One or more essential Azure OpenAI configuration values are missing in config.yaml.")

 
    if not api_key:
        raise ValueError(f"Azure OpenAI API key environment variable '{api_key_env_var}' not set. Please set it before running.")
    
    print(f"Initializing AzureChatOpenAI client with deployment: {deployment_name}")
    return AzureChatOpenAI(
        azure_deployment=deployment_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens
    )

def generate_llm_prompt(record_pair_data: dict) -> list:
    """Generates the messages list for the LLM prompt."""
    prompt_content = f"""
    You are an expert in vendor record matching. Your task is to review two vendor records
    and their associated matching scores, then provide a binary verdict ('match' or 'no_match')
    along with a concise explanation for your decision.

    Consider the following information:

    Record 1:
    {json.dumps(record_pair_data.get('record1', {}), indent=2)}

    Record 2:
    {json.dumps(record_pair_data.get('record2', {}), indent=2)}

    Matching Scores:
    {json.dumps(record_pair_data.get('scores', {}), indent=2)}

    Metadata:
    {json.dumps(record_pair_data.get('metadata', {}), indent=2)}

    Key score indicators:
    - `final_score`: A combined score from multiple matchers. High values (e.g., > 0.85) usually indicate a strong match.
    - `fuzzy_score`: Similarity based on fuzzy matching.

    Based on all the provided evidence, make a verdict.
    Return your decision in a JSON format with two fields:
    - `llm_verdict`: "match" or "no_match"
    - `llm_explanation": "A brief, clear explanation of why you made that decision."

    Example Output:
    ```json
    {{
      "llm_verdict": "match",
      "llm_explanation": "The vendor names are strong phonetic matches, address components are nearly identical, and ensemble score is high. Likely the same entity."
    }}
    ```
    Always enclose your JSON output within ```json and ``` markdown fences.
    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_CONTENT),
        HumanMessage(content=prompt_content),
    ]
    return messages

def call_llm_and_parse_response(llm_client: AzureChatOpenAI, record_pair_data: dict, log_raw: bool = False) -> dict:
    """
    Calls the LLM with the generated prompt and robustly parses its JSON response.
    Returns a dictionary of the LLM's verdict and explanation.
    """
    llm_output_str = "" 
    try:
        messages = generate_llm_prompt(record_pair_data)
        response = llm_client.invoke(messages)
        llm_output_str = response.content.strip()

        if log_raw:
            record_id_1 = record_pair_data.get('metadata', {}).get('record1_id', 'N/A')
            record_id_2 = record_pair_data.get('metadata', {}).get('record2_id', 'N/A')
            print(f"\n--- Raw LLM Output for record {record_id_1}-{record_id_2} ---")
            print(llm_output_str)
            print("-------------------------------------------\n")

        json_match = re.search(r"```json\s*(.*?)\s*```", llm_output_str, re.DOTALL)
        if json_match:
            json_payload = json_match.group(1)
        else:
            json_payload = llm_output_str

        llm_result = json.loads(json_payload)

        if "llm_verdict" not in llm_result or "llm_explanation" not in llm_result:
            raise ValueError(f"LLM response missing required keys: {llm_output_str}")

        return llm_result

    except json.JSONDecodeError as jde:
        print(f"JSON Decoding Error for input: {json.dumps(record_pair_data)}. Raw LLM output: '{llm_output_str}'. Error: {jde}")
        return {
            "llm_verdict": "error",
            "llm_explanation": f"LLM returned invalid JSON: {jde}. Raw output: '{llm_output_str[:200]}...'"
        }
    except Exception as e:
        print(f"General error processing record_pair_data: {json.dumps(record_pair_data)}. Error: {e}. Raw LLM output: '{llm_output_str}'")
        return {
            "llm_verdict": "error",
            "llm_explanation": f"LLM call failed: {e}. Raw output: '{llm_output_str[:200]}...'"
        }

def read_jsonl_file(file_path: str):
    """Reads a JSONL file line by line and yields parsed JSON objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The input file '{file_path}' was not found. "
                                "Please ensure it exists.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: '{line}' - Error: {e}")


def run_llm_validation(config: dict):
    try:
        # 0) Check the postprocessing toggle for LLM validation
        post_cfg = config.get("postprocessing", {})
        llm_enabled = post_cfg.get("llm_validation", False)
        if not llm_enabled:
            return []

        # 1) Extract specific sections
        azure_openai_config   = config.get("azure_openai", {})
        llm_validation_config = config.get("llm_validation", {})
        input_data_config     = config.get("input_data", {})

        # Initialize LLM client using the extracted Azure OpenAI configurations
        llm_client = initialize_llm_client(azure_openai_config)

        # Get the input file path from the config
        input_file_path = input_data_config.get("file_path")
        if not input_file_path:
            raise ValueError("Input file path not specified in config.yaml under 'input_data.file_path'.")

        print(f"--- Starting LLM Validation from '{input_file_path}' (Azure OpenAI) ---")
        # Show the postprocessing toggle rather than looking for "enabled" under top-level
        print(f"LLM validation enabled (postprocessing): {llm_enabled}")
        print(f"Auto-match threshold: {llm_validation_config.get('auto_match_threshold', 'N/A')}")
        print(f"Auto-non-match threshold: {llm_validation_config.get('auto_non_match_threshold', 'N/A')}")
        print(f"LLM call threshold: {llm_validation_config.get('confidence_threshold_for_llm', 'N/A')}\n")

        final_validated_records = []
        # Read data from the JSONL file
        for i, record_pair in enumerate(read_jsonl_file(input_file_path)):
            record_id_1 = record_pair.get("metadata", {}).get("record1_id", "N/A")
            record_id_2 = record_pair.get("metadata", {}).get("record2_id", "N/A")
            print(f"Processing record pair {i+1} (IDs: {record_id_1}, {record_id_2})...")

            llm_verdict = "N/A"
            llm_explanation = "LLM not called."
            
            # Extract the relevant score for decision making
            final_score = record_pair.get("scores", {}).get("final_score", 0.0)

            if final_score >= llm_validation_config.get("auto_match_threshold", 1.0):
                llm_verdict = "auto_match"
                llm_explanation = (
                    f"Final score ({final_score:.4f}) is above auto-match threshold "
                    f"({llm_validation_config.get('auto_match_threshold', 1.0):.2f})."
                )
            elif final_score <= llm_validation_config.get("auto_non_match_threshold", 0.0):
                llm_verdict = "auto_no_match"
                llm_explanation = (
                    f"Final score ({final_score:.4f}) is below auto-non-match threshold "
                    f"({llm_validation_config.get('auto_non_match_threshold', 0.0):.2f})."
                )
            elif final_score >= llm_validation_config.get("confidence_threshold_for_llm", 0.0):
                # Call LLM only if within the "ambiguous" range and above LLM call threshold
                llm_output = call_llm_and_parse_response(
                    llm_client, 
                    record_pair, 
                    log_raw=llm_validation_config.get("log_raw_llm_response", False)
                )
                llm_verdict = llm_output.get("llm_verdict", "error")
                llm_explanation = llm_output.get("llm_explanation", "Error in processing.")
            else:
                llm_verdict = "skipped"
                llm_explanation = (
                    f"Final score ({final_score:.4f}) is below LLM call threshold "
                    f"({llm_validation_config.get('confidence_threshold_for_llm', 0.0):.2f})."
                )
            
            validated_record = {
                "record1_data": record_pair.get("record1", {}),
                "record2_data": record_pair.get("record2", {}),
                "scores": record_pair.get("scores", {}),
                "metadata": record_pair.get("metadata", {}),
                "llm_verdict": llm_verdict,
                "llm_explanation": llm_explanation
            }
            final_validated_records.append(validated_record)

        return final_validated_records

    except FileNotFoundError as fnfe:
        print(f"\nError: {fnfe}")
        print("Please ensure your 'resolver_config/config.yaml' file exists and its paths are correct,")
        print("and that 'postprocessing/for_llm.json' also exists.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
    return []


    