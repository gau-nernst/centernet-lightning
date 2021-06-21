from typing import Union, Dict
import os
import yaml

from .centernet import CenterNet

def build_centernet_from_cfg(cfg_file: Union[str, Dict]):
    """Build CenterNet from a confile file.

    Args:
        cfg_file (str or dict): Config file to build CenterNet, either path to the config file, or a config file dictionary
    """
    if type(cfg_file) == str:
        assert os.path.exists(cfg_file), "Config file does not exist"
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            params = config["model"]

    else:
        params = cfg_file

    model = CenterNet(**params)
    return model
