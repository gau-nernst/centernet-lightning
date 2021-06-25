import os
import yaml
from copy import deepcopy

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if "__base__" in config:
        config_dir, _ = os.path.split(config_path)
        base_config_path = os.path.join(config_dir, config["__base__"])

        base_config = load_config(base_config_path)

        base_config = update_dict(base_config, config)

        return base_config
    
    return config

def update_dict(dict1, dict2):
    dict1 = deepcopy(dict1)
    dict2 = deepcopy(dict2)

    for k, v in dict2.items():
        # dict2 has key that dict1 doesn't have, just copy over
        if k not in dict1:
            dict1[k] = deepcopy(v)
        
        # the value in dict1 is a dictionary, call the function recursively
        elif isinstance(dict1[k], dict):
            dict1[k] = update_dict(dict1[k], v)
        
        # the value in dict1 is not a dictionary, just copy values from dict2
        else:
            dict1[k] = deepcopy(v)

    return dict1
