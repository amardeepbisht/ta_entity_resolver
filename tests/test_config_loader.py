from config.config_loader import load_config

def main():
    config = load_config("config/config.yaml")
    print("Config Loaded Successfully:")
    print(config)

if __name__ == "__main__":
    main()
