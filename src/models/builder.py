from typing import Union, Dict
import os

from .centernet import CenterNet
from ..utils import load_config

def build_centernet_from_cfg(cfg_file: Union[str, Dict]):
    """Build CenterNet from a confile file.

    Args:
        cfg_file (str or dict): Config file to build CenterNet, either path to the config file, or a config file dictionary
    """
    if type(cfg_file) == str:
        assert os.path.exists(cfg_file), "Config file does not exist"
        config = load_config(cfg_file)
        params = config["model"]

    else:
        params = cfg_file

    model = CenterNet(**params)
    return model
