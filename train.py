import os
import warnings
from typing import Dict, Union
import yaml

from dotenv import load_dotenv
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets import COCODataModule
from model import build_centernet_from_cfg
from utils import LogImageCallback

def train(config: Union[str, Dict]):
    if type(config) == str:
        assert os.path.exists(config), f"{config} does not exist"
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    model = build_centernet_from_cfg(config["model"])
    coco_datamodule = COCODataModule(**config["data"])
    
    if "logger" in config["trainer"]:
        if config["trainer"]["logger"]["name"] == "wandb":
            logger = WandbLogger(**config["trainer"]["logger"]["params"])
            logger.watch(model)
        
        elif config["trainer"]["logger"]["name"] == "tensorboard":
            logger = TensorBoardLogger(**config["trainer"]["logger"]["params"])

        else:
            warnings.warn(f'{config["trainer"]["logger"]["name"]} is not supported. Using default Lightning logger (tensorboard)')
            logger = True

        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            LogImageCallback(config["trainer"]["logger"]["name"])
        ]
     
    else:
        logger = True
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            LogImageCallback("tensorboard")
        ]

    trainer = pl.Trainer(
        **config["trainer"]["params"],
        logger=logger,
        callbacks=callbacks
    )
   
    trainer.fit(model, coco_datamodule)

if __name__ == "__main__":
    load_dotenv(override=True)      # load wandb API key

    parser = argparse.ArgumentParser(description="Train CenterNet from a config file")
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()

    train(args.config)
