from typing import Dict

import torch
from torch import nn

from ..losses import CornerNetFocalLossWithLogits, QualityFocalLossWithLogits

def _make_output_head(in_channels: int, hidden_channels: int, out_channels: int, init_bias: float = None):
    # Reference implementations
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L125    use num_filters = 256
    # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py#L5      use num_filters = in_channels
    conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
    relu = nn.ReLU(inplace=True)
    conv2 = nn.Conv2d(hidden_channels, out_channels, 1)

    if init_bias is not None:
        conv2.bias.data.fill_(init_bias)

    output_head = nn.Sequential(conv1, relu, conv2)
    return output_head

class HeatmapHead(nn.Module):
    _loss_mapper = {
        "cornernet_focal": CornerNetFocalLossWithLogits,
        "quality_focal": QualityFocalLossWithLogits
    }

    def __init__(self, in_channels, num_classes, init_bias=-2.19, target_method="cornernet", loss_function="cornernet_focal", loss_weight=1):
        super().__init__()
        self.head = _make_output_head(in_channels, in_channels, num_classes, init_bias=init_bias)
        self.target_method = target_method
        self.loss_function = self._loss_mapper[loss_function]()
        self.loss_weight = loss_weight
    
    def forward(self, x):
        out = self.head(x)
        return out

    def compute_loss(self, pred, target):
        # render target heatmap
        heatmap = pred["heatmap"]
        return loss

class Box2DHead(nn.Module):
    _loss_mapper = {
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "iou": None,
        "giou": None,
        "diou": None,
        "ciou": None
    }
    out_channels = 4

    def __init__(self, in_channels, init_bias=None, loss_function="l1", loss_weight=1):
        super().__init__()
        self.head = _make_output_head(in_channels, in_channels, self.out_channels, init_bias=init_bias)
        self.loss_function = self._loss_mapper[loss_function](reduction="none")
        self.loss_weight = loss_weight

    def forward(self, x):
        out = self.head(x)
        return out

    def compute_loss(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        pred_box_map = pred["box_2d"]
        target_box = target["bboxes"]
        mask = target["mask"]
        batch_size, channels, output_height, output_width = pred_box_map.shape

        # scale up to feature map size and convert to integer        
        x_indices = target_box[...,0].clone() * output_width
        y_indices = target_box[...,1].clone() * output_height
        xy_indices = y_indices.long() * output_width + x_indices.long()
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size, channels, -1))

        pred_box_map = pred_box_map.view(batch_size, channels, -1)
        pred_box_offset = torch.gather(pred_box_map, dim=-1, index=xy_indices)
        pred_box = torch.zeros_like(pred_box_offset, device=pred_box_offset.device)
        pred_box[...,0] = x_indices - pred_box_offset[...,0]
        pred_box[...,1] = y_indices + pred_box_offset[...,1]
        pred_box[...,2] = x_indices - pred_box_offset[...,2]
        pred_box[...,3] = y_indices + pred_box_offset[...,3]

        

class TimeDisplacementHead:
    pass