from typing import Callable, Dict, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from vision_toolbox import backbones
from vision_toolbox.components import ConvBnAct


class BasicHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        filters_list: Iterable[int],
        block: Callable[[int, int], nn.Module] = ConvBnAct,
        init_bias: float = None,
    ):
        assert len(filters_list) > 0
        super().__init__()
        for i, channels in enumerate(filters_list[:-1]):
            self.add_module(f"block_{i+1}", block(in_channels, channels))
            in_channels = channels

        self.out_conv = nn.Conv2d(in_channels, filters_list[-1], 1)
        if init_bias is not None:
            self.out_conv.bias.data.fill_(init_bias)


class BasicModel(nn.Module):
    def __init__(
        self,
        backbone: backbones.BaseBackbone,
        neck: nn.Module,
        heads: nn.ModuleDict,
        num_feature_maps: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.heads = heads
        self.num_feature_maps = num_feature_maps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone.get_feature_maps(x)[-self.num_feature_maps :]
        out = self.neck(out)[-1]
        outputs = {}
        for head_name, head_module in self.heads.items():
            outputs[head_name] = head_module(out)
        return outputs


class CosineWithWarmup(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_epochs: int,
        warmup_epochs: int,
        warmup_decay: float,
    ):
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
        warmup_scheduler = LinearLR(
            optimizer, start_factor=warmup_decay, total_iters=warmup_epochs
        )
        super().__init__(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        # https://github.com/pytorch/pytorch/issues/67318
        if not hasattr(self, "optimizer"):
            setattr(self, "optimizer", optimizer)
