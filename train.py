import os
from typing import Dict, Union
import argparse
from copy import deepcopy
from functools import partial

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks

try:
    import wandb
except ImportError:
    pass

from centernet_lightning.models import build_centernet
from centernet_lightning.datasets import build_dataloader
from centernet_lightning.utils import LogImageCallback, load_config

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

    logger_mapper = pl_loggers.__dict__
    callback_mapper = {**pl_callbacks.__dict__, "LogImageCallback": partial(LogImageCallback, config["data"]["validation"]["dataset"])}

    trainer_config = parse_trainer_config(config["trainer"], logger_mapper, callback_mapper)
    trainer_config["logger"].log_hyperparams({"data": config["data"], "trainer": config["trainer"]})
    if isinstance(trainer_config["logger"], pl_loggers.WandbLogger):
        trainer_config["logger"].experiment.watch(model, log="all", log_graph=True)
    

    trainer = pl.Trainer(**trainer_config)

    trainer.fit(model, train_dataloader, val_dataloader)
    if isinstance(trainer_config["logger"], pl_loggers.WandbLogger):
        wandb.finish()

def parse_trainer_config(trainer_cfg, logger_mapper, callback_mapper):
    config = deepcopy(trainer_cfg)

    # parse logger
    if "logger" in config:
        logger_config = config["logger"]
        logger = logger_mapper[logger_config["name"]](**logger_config["params"])
    else:
        save_dir = os.getcwd()
        name = "lightning_logs"
        log_path = os.path.join(save_dir, name)
        version = len(os.listdir(log_path))+1 if os.path.exists(log_path) else 1

        logger = pl_loggers.TensorBoardLogger(save_dir=save_dir, version=version, name=name, log_graph=True)
    config["logger"] = logger

    # parse callbacks
    if "callbacks" in config:
        callbacks = []
        for callback_config in config["callbacks"]:
            item = callback_mapper[callback_config["name"]](**callback_config["params"])
            callbacks.append(item)
    else:
        callbacks = None
    config["callbacks"] = callbacks
    
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CenterNet from a config file")
    parser.add_argument("--config", type=str, help="path to config file")
    args = parser.parse_args()

    train(args.config)
