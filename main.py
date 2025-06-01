from config.config_loader import load_config
from data_loader.load_data import load_input_data
from preprocessor.preprocessor import preprocess_data


if __name__ == "__main__":
    config = load_config("config/config.yaml")

    input_cfg = config["input"]
    file_path = input_cfg["path"]
    file_format = input_cfg.get("format", "csv")
    sheet_name = input_cfg.get("sheet_name", None)

    engine = config.get("engine", "pandas")

    df = load_input_data(file_path=file_path, engine=engine, file_format=file_format, sheet_name=sheet_name)
    print(df.head())
    print("Available columns:", df.columns.tolist())
    print("Match columns from config:", config.get("match_columns", []))


    df2 = preprocess_data(df, config)

    print("Preprocessing complete. Sample output:")
    print(df2.head())