import yaml

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration data.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config