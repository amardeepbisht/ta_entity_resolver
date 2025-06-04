import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def write_json_to_csv(data: list, output_path: str):
    """
    Flattens a list of nested JSON/dict records and writes them to a CSV file.

    Parameters:
    - data (list): List of dictionaries (e.g., LLM validation results)
    - output_path (str): Path to write the CSV file
    """
    try:
        if not data:
            logger.warning("No data provided to write to CSV.")
            return

        df = pd.json_normalize(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully wrote flattened CSV to: {output_path}")
    except Exception as e:
        logger.error(f"Error writing CSV to {output_path}: {e}", exc_info=True)
