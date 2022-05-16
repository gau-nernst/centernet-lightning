import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from centernet_lightning.models.basics import BasicModel
import torch
from centernet_lightning.models.builder import build_model
# from centernet_lightning.datasets.builder import build_datamodules
import pytorch_lightning as pl

# from centernet_lightning.models.centernet import CenterNet


@hydra.main(config_path="configs", config_name="base")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model = build_model(cfg.model)
    train_dataloader = build_dataloader(cfg.data_train)
    val_dataloader = build_dataloader(cfg.data_val)

    trainer = hydra.utils.instantiate(cfg.trainer, _target_="pytorch_lightning.Trainer")

    # trainer = pl.Trainer()
    # trainer.fit()


if __name__ == "__main__":
    main()
