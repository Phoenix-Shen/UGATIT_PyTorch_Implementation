# %%
import yaml
import os


def load_config(config_file_path: str) -> dict:
    """
    load config from a yaml file and returns a dictionary that contains the configuration and print them 
    --------
    params
    config_file_path: your config file path, normally is the config.yaml, you should not change the name
    """
    # check the file location
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            "the file does not exist, please check the path")

    with open(config_file_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), yaml.FullLoader)

    print("###################YOUR SETTINGS###################")
    for key in args.keys():
        print(f"[{key}]".ljust(30, " "), f"--->{args[key]}")

    create_folder(args)
    check_config(args)


def create_folder(config: dict) -> None:
    """
    Create a directory to store training results
    --------
    params: config, a dictionary that stores the settings
    """
    result_path = os.path.join(config["result_dir"], config["dataset"])
    if not os.path.exists(result_path):
        os.makedirs(os.path.join(result_path, "img"))
        os.makedirs(os.path.join(result_path, "model"))
        os.makedirs(os.path.join(result_path, "test"))


def check_config(config: dict) -> bool:
    """
    Check for incorrect parameters
    """
    assert config["batch_size"] >= 1
    assert config["iteration"] >= 1
    assert config["n_res"] >= 1
    assert config["n_diss"] >= 1
    assert config["ch"] >= 1
    assert len(config["dataset"]) > 0
