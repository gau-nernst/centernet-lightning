from typing import Any, Dict, List, Union
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

# Reference implementations
# https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py     num_filters = 256
# https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py                                     num_filters = in_channels
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
    def __init__(self, backbone: BaseBackbone, neck: BaseNeck, heads: nn.Module, extra_block=None):
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


class MetaCenterNet(pl.LightningModule):
    """Meta architecture for CenterNet. Implement training logic
    """
    def __init__(
        self,
        # model
        backbone: BaseBackbone,
        neck: BaseNeck,
        heads: nn.Module,
        extra_block: nn.Module=None,

        # optimizer and scheduler
        optimizer: str="SGD",
        lr: float=0.05,
        weight_decay: float=2e-5,
        norm_weight_decay: float=0,
        warmup_epochs: int=5,
        warmup_decay: float=0.01,
        
        # data
        # batch_size: int=8,
        # num_workers: int=2,
        # train_data: Dict[str, Any]=None,
        # val_data: Dict[str, Any]=None,

        jit: bool=False
    ):
        super().__init__()
        # self.backbone = backbone
        # self.extra_block = extra_block
        # self.neck = neck
        # self.heads = nn.ModuleDict(heads)
        self.model = GenericModel(backbone, neck, heads, extra_block=extra_block)
        if jit:
            self.model = torch.jit.script(self.model)

    def get_output_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """Return encoded outputs, a dict of output feature maps. Use this output to either compute loss or decode to detections. Heatmap is before sigmoid
    #     """
    #     feat = self.backbone.forward_features(x)
    #     if self.extra_block is not None:        # e.g. SPP
    #         feat[-1] = self.extra_block(feat[-1])
        
    #     feat = self.neck(feat)
    #     outputs = {name: module(feat) for name, module in self.heads.items()}
    #     return outputs
        return self.model(x)

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Union[List, int]]]) -> Dict[str, torch.Tensor]:
        pass
    #     """Return a dict of losses for each output head, and weighted total loss. This method is called during the training step
    #     """
    #     losses = {"total": torch.tensor(0., device=self.device)}
    #     for name, module in self.heads.items():
    #         module: BaseHead
    #         losses[name] = module.compute_loss(outputs, targets)
    #         losses["total"] += losses[name] * module.loss_weight

    #     return losses

    def training_step(self, batch, batch_idx):
        images, targets = batch
        encoded_outputs = self.get_output_dict(images)
        losses = self.compute_loss(encoded_outputs, targets)
        for k, v in losses.items():
            self.log(f"train/{k}_loss", v)

        return losses["total"]

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
