from typing import Dict, List, Union
from functools import partial

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl

from vision_toolbox.backbones import BaseBackbone
from vision_toolbox.necks import BaseNeck
from vision_toolbox.components import ConvBnAct


_optimizers = {
    "SGD": partial(torch.optim.SGD, momentum=0.9),
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": partial(torch.optim.RMSprop, momentum=0.9)
}


class GenericHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, width: int=256, depth: int=1, block=ConvBnAct, init_bias: float=None):
        super().__init__()
        for i in range(depth):
            in_c = in_channels if i == 0 else width
            self.add_module(f"block_{i+1}", block(in_c, width))

        self.out_conv = nn.Conv2d(width, out_channels, 1)
        if init_bias is not None:
            self.out_conv.bias.data.fill_(init_bias)


class GenericModel(nn.Module):
    def __init__(self, backbone: BaseBackbone, neck: BaseNeck, heads: nn.Module, extra_block: nn.Module=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads
        self.extra_block = extra_block
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone.forward_features(x)
        out = self.neck(out)
        if self.extra_block is not None:        # e.g. SPP
            out = self.extra_block(out)
        out = {name: head(out) for name, head in self.heads.named_children()}
        return out


class GenericLightning(pl.LightningModule):
    def __init__(
        self,
        # model
        backbone: BaseBackbone,
        neck: BaseNeck,
        heads: nn.Module,
        extra_block: nn.Module=None,
        jit: bool=False,

        # optimizer and scheduler
        optimizer: str="SGD",
        lr: float=0.05,
        weight_decay: float=2e-5,
        norm_weight_decay: float=0,
        warmup_epochs: int=5,
        warmup_decay: float=0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GenericModel(backbone, neck, heads, extra_block=extra_block)
        if jit:
            self.model = torch.jit.script(self.model)

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Union[List, int]]]) -> Dict[str, torch.Tensor]:
        pass

    def get_dataloader(self, train=True):
        pass

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            length = max([len(name) for name, _ in self.model.named_children()]) + 1
            for name, module in self.model.named_children():
                num_params = sum([x.numel() for x in module.parameters()]) / 1e6
                print(f'{name:{length}}: {num_params:.1f}M')

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        losses = self.compute_loss(outputs, targets)
        for k, v in losses.items():
            self.log(f"train/{k}_loss", v)

        return losses["total"]

    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def configure_optimizers(self):
        # norm's weight decay = 0
        # https://github.com/pytorch/vision/blob/main/torchvision/ops/_utils.py
        if self.hparams.norm_weight_decay is not None:
            norm_classes = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)
            norm_params, other_params = [], []
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
            
            if not hasattr(lr_scheduler, "optimizer"):      # https://github.com/pytorch/pytorch/issues/67318
                setattr(lr_scheduler, "optimizer", optimizer)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
        }
