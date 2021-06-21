import os
import warnings
from typing import Dict, Union
import yaml

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.datasets import COCODataModule
from src.models import build_centernet_from_cfg
from src.utils import LogImageCallback

def train(config: Union[str, Dict]):
    """Train CenterNet from a config.

    Args
        config (str or dict): Either path to a config file or a config dictionary
    """
    if type(config) == str:
        assert os.path.exists(config), f"{config} does not exist"
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    model = build_centernet_from_cfg(config["model"])
    coco_datamodule = COCODataModule(**config["data"])
    coco_datamodule.prepare_data()
    
    if "logger" in config["trainer"]:
        logger_name = config["trainer"]["logger"]["name"]

        if logger_name == "wandb":
            logger = WandbLogger(**config["trainer"]["logger"]["params"])
            logger.watch(model)
        
        elif logger_name == "tensorboard":
            logger = TensorBoardLogger(**config["trainer"]["logger"]["params"])

        else:
            warnings.warn(f'{logger_name} is not supported. Using default Lightning logger (tensorboard)')
            logger = True
            logger_name = "tensorboard"

    else:
        logger = True
        logger_name = "tensorboard"

    trainer = pl.Trainer(
        **config["trainer"]["params"],
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            LogImageCallback("datasets/COCO/val2017", "datasets/COCO/val2017/detections.pkl")
        ]
    )
   
    trainer.fit(model, coco_datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CenterNet from a config file")
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()

    train(args.config)
