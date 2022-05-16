from typing import List, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from .basics import BasicModel


class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        loss_cfg: DictConfig,
        decoder_cfg: DictConfig,
        evaluator_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_cfg)
        self.loss = hydra.utils.instantiate(loss_cfg)
        self.decoder = hydra.utils.instantiate(decoder_cfg)
        self.evaluator = hydra.utils.instantiate(evaluator_cfg)

    def training_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["target"]
        outputs = self.model(images)
        
        loss = self.loss(outputs, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch["image"], batch["target"]
        outputs = self.model(images)

        predictions = self.decoder(outputs)
        metrics = self.evaluator(predictions, targets)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams.optimizer_cfg, self.model)
        lr_scheduler = hydra.utils.instantiate(self.hparams.scheduler_cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def build_model(model_cfg: DictConfig) -> BasicModel:
    n = model_cfg.num_feature_maps
    backbone = hydra.utils.instantiate(model_cfg.backbone)
    neck = hydra.utils.instantiate(model_cfg.neck, in_channels_list=backbone.out_channels_list[-n:])

    heads = nn.ModuleDict()
    for head_name, head_cfg in model_cfg.heads.items():
        heads[head_name] = hydra.utils.instantiate(head_cfg, in_channels=neck.out_channels)

    model = BasicModel(backbone, neck, heads, num_feature_maps=n)
    return model


def build_optimizer(optimizer_cfg: DictConfig, model: nn.Module):
    param_groups = _create_param_groups(model)
    optimizer = hydra.utils.instantiate(optimizer_cfg, parameters=param_groups)
    return optimizer


def _create_param_groups(model: nn.Module):
    base_classes = (nn.Linear, nn.modules.conv._ConvNd)
    norm_classes = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)

    wd_params: List[torch.Tensor] = []
    no_wd_params: List[torch.Tensor] = []

    def process_module(module: nn.Module):
        if isinstance(module, base_classes):
            wd_params.append(module.weight)
            no_wd_params.append(module.bias)

        elif isinstance(module, norm_classes):
            no_wd_params.append(module.weight)
            no_wd_params.append(module.bias)
        
        else:
            wd_params.extend(module.parameters(recurse=False))
            for child_module in module.children():
                process_module(child_module)
    
    process_module(model)

    def is_valid_param(p: Optional[torch.Tensor]):
        return p is not None and p.requires_grad
    
    wd_params = [p for p in wd_params if is_valid_param(p)]
    no_wd_params = [p for p in no_wd_params if is_valid_param(p)]

    num_total_params = len([p for p in model.parameters() if p.requires_grad])
    num_total_params_groups = len(wd_params) + len(no_wd_params)
    assert num_total_params == num_total_params_groups, (num_total_params, num_total_params_groups)
    
    param_groups = [
        {"params": wd_params},
        {"params": no_wd_params, "weight_decay": 0}
    ]
    return param_groups
