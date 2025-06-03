# resolver_config/config_loader.py

import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration data.
    """
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:   
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as ye:
        logger.error(f"Error parsing YAML file: {ye}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading config: {e}")
        raise
