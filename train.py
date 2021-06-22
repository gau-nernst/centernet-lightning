import os
import warnings
from typing import Dict, Union
import yaml

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models import build_centernet_from_cfg
from src.datasets.builder import build_dataloader
from src.utils import LogImageCallback

def train(config: Union[str, Dict]):
    """Train CenterNet from a config.

    Args
        config (str or dict): Either path to a config file or a config dictionary
    """
    # load config file
    if type(config) == str:
        assert os.path.exists(config), f"{config} does not exist"
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    # build model and dataset
    model = build_centernet_from_cfg(config["model"])
    train_dataloader = build_dataloader(model, **config["data"]["train"])
    val_dataloader = build_dataloader(model, **config["data"]["validation"])

    logger = parse_logger_config(config["trainer"]["logger"], model) if "logger" in config["trainer"] else True

    trainer = pl.Trainer(
        **config["trainer"]["params"],
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            LogImageCallback(config["data"]["validation"])
        ]
    )
   
    trainer.fit(model, train_dataloader, val_dataloader)

def parse_logger_config(logger_cfg, model):
    logger_name = logger_cfg["name"]

    if logger_name == "wandb":
        logger = WandbLogger(**logger_cfg["params"])
        logger.watch(model)
    
    elif logger_name == "tensorboard":
        logger = TensorBoardLogger(**logger_cfg["params"])

    else:
        warnings.warn(f'{logger_name} is not supported. Using default Lightning logger (tensorboard)')
        logger = True
    
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CenterNet from a config file")
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()

    train(args.config)
