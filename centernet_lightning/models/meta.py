from typing import Callable, Dict, List, Union
from functools import partial

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
try:
    import wandb
except ImportError:
    wandb = None

from vision_toolbox.backbones.base import BaseBackbone


_optimizers = {
    "SGD": partial(torch.optim.SGD, momentum=0.9),
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": partial(torch.optim.RMSprop, momentum=0.9)
}


class BaseHead(nn.Module):
    # Reference implementations
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py     num_filters = 256
    # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py                                     num_filters = in_channels
    def __init__(self, in_channels: int, out_channels: int, width: int=256, depth: int=1, init_bias: float=None, loss_weight: float=1.):
        super().__init__()
        self.loss_weight = loss_weight
        layers = []
        for i in range(depth):
            in_c = in_channels if i == 0 else width
            layers.append(nn.Conv2d(in_c, width, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))            
            nn.init.kaiming_normal_(layers[-2].weight, mode="fan_out", nonlinearity="relu")
        
        layers.append(nn.Conv2d(width, out_channels, 1))
        if init_bias is not None:
            layers[-1].bias.data.fill_(init_bias)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def compute_loss(self, pred, target, eps=1e-8):
        pass


class MetaCenterNet(pl.LightningModule):
    """Meta architecture for CenterNet
    """
    def __init__(
        self,
        backbone: BaseBackbone,
        neck: nn.Module,
        heads: Dict[str, Callable[..., BaseHead]],

        head_width: int=256,
        head_depth: int=1,

        # optimizer and scheduler
        optimizer: str="SGD",
        lr: float=0.05,
        weight_decay: float=2e-5,
        norm_weight_decay: float=0,
        warmup_epochs: int=5,
        warmup_decay: float=0.01,
        
        # logging
        log_feat_map: bool=False,
        log_values_hist: bool=False,
        log_freq: int=500
        ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone 
        self.neck = neck

        self.stride = self.backbone.stride // self.neck.stride
        self.heads = nn.ModuleDict({k: v(neck.out_channels, stride=self.stride, width=head_width, depth=head_depth) for k, v in heads.items()})

    def get_encoded_outputs(self, x: torch.Tensor, include_feat_map: bool=True) -> Dict[str, torch.Tensor]:
        """Return encoded outputs, a dict of output feature maps. Use this output to either compute loss or decode to detections. Heatmap is before sigmoid
        """
        feat = self.backbone.forward_features(x)
        feat = self.neck(feat)
        outputs = {name: module(feat) for name, module in self.heads.items()}
        if self.hparams.log_feat_map and include_feat_map:
            outputs["features"] = feat
        
        return outputs

    def compute_loss(self, preds: Dict[str, torch.Tensor], targets: List[Dict[str, Union[List, int]]]):
        """Return a dict of losses for each output head, and weighted total loss. This method is called during the training step
        """
        losses = {"total": torch.tensor(0., device=self.device)}
        for name, module in self.heads.items():
            module: BaseHead
            losses[name] = module.compute_loss(preds, targets)
            losses["total"] += losses[name] * module.loss_weight

        return losses

    def training_step(self, batch, batch_idx):
        images, targets = batch
        encoded_outputs = self.get_encoded_outputs(images)
        losses = self.compute_loss(encoded_outputs, targets)
        for k, v in losses.items():
            self.log(f"train/{k}_loss", v)

        if self.hparams.log_values_hist:
            for k, v in encoded_outputs.items():
                self.log_histogram(f"output_values/{k}", v)

        return losses["total"]

    def log_histogram(self, name: str, values: torch.Tensor):
        """Log histogram. Only TensorBoard and Wandb are supported
        """
        if self.trainer.global_step == 0 or (self.trainer.global_step + 1) % self.hparams.log_freq == 0:
            flatten_values = values.detach().view(-1).cpu().float().numpy()

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_histogram(name, flatten_values, global_step=self.global_step)
            elif isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({name: wandb.Histogram(flatten_values), "global_step": self.global_step})

    def configure_optimizers(self):
        if self.hparams.norm_weight_decay is not None:      # norm's weight decay = 0
            # https://github.com/pytorch/vision/blob/main/torchvision/ops/_utils.py
            norm_classes = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)
            
            norm_params = []
            other_params = []
            for module in self.modules():
                if next(module.children(), None):
                    other_params.extend(p for p in module.parameters(recurse=False) if p.requires_grad)
                elif isinstance(module, norm_classes):
                    norm_params.extend(p for p in module.parameters() if p.requires_grad)
                else:
                    other_params.extend(p for p in module.parameters() if p.requires_grad)

            param_groups = (norm_params, other_params)
            wd_groups = (self.hparams.norm_weight_decay, self.hparams.weight_decay)
            parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

        else:
            parameters = self.parameters()

        optimizer = _optimizers[self.hparams.optimizer](parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs-self.hparams.warmup_epochs)
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=self.hparams.warmup_decay, total_iters=self.hparams.warmup_epochs)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[self.hparams.warmup_epochs])
            
            # https://github.com/pytorch/pytorch/issues/67318
            if not hasattr(lr_scheduler, "optimizer"):
                setattr(lr_scheduler, "optimizer", optimizer)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
        }
