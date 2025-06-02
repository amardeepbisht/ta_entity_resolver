from config.config_loader import load_config
from data_loader.load_data import load_input_data
from preprocessor.preprocessor import preprocess_data


if __name__ == "__main__":
    config = load_config("config/config.yaml")

    input_cfg = config["input"]
    engine = config.get("engine", "pandas")

    # Simply pass the input_cfg dict now
    df = load_input_data(file_config=input_cfg, engine=engine)
    
    print(df.head())
    print("Available columns:", df.columns.tolist())
    print("Match columns from config:", config.get("match_columns", []))

    df2 = preprocess_data(df, config)

    print("Preprocessing complete. Sample output:")
    print(df2.head())


  