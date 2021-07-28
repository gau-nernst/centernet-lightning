import warnings
from typing import Dict, Union
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb

from src.models import build_centernet
from src.datasets import build_dataloader
from src.utils import LogImageCallback, load_config

def train(config: Union[str, Dict]):
    """Train CenterNet from a config.

    Args
        config (str or dict): Either path to a config file or a config dictionary
    """
    # load config file
    if isinstance(config, str):
        config = load_config(config)

    # build model and dataset
    model = build_centernet(config["model"])
    train_dataloader = build_dataloader(config["data"]["train"])
    val_dataloader = build_dataloader(config["data"]["validation"])

    logger = parse_logger_config(config["logger"], model) if "logger" in config else True

    trainer = pl.Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="val/total_loss", save_last=True),
            LearningRateMonitor(logging_interval="step"),
            LogImageCallback(config["data"]["validation"]["dataset"], n_epochs=5)
        ]
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

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
