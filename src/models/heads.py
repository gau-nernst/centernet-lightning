from typing import Dict, Union

import numpy as np
import torch
from torch import nn

from ..losses import CornerNetFocalLossWithLogits, QualityFocalLossWithLogits
from ..utils import convert_cxcywh_to_x1y1x2y2, load_config

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
        self.num_classes = num_classes
        self.target_method = target_method
        self.loss_function = self._loss_mapper[loss_function]()
        self.loss_weight = loss_weight

    def forward(self, x):
        out = self.head(x)
        return out

    def compute_loss(self, pred, target, eps=1e-8):
        # overall steps
        # 1. calculate target heatmap
        # 2. apply loss
        heatmap = pred["heatmap"]
        bboxes = target["bboxes"]
        labels = target["labels"]
        mask = target["mask"]

        batch_size = heatmap.shape[0]
        target_heatmap = torch.zeros_like(heatmap, device=heatmap.device)
        # target_heatmap = target["target_heatmap"]

        for b in range(batch_size):
            self._render_target_heatmap(target_heatmap[b], bboxes[b], labels[b], mask[b])

        loss = self.loss_function(heatmap, target_heatmap) / (mask.sum() + eps)

        return loss

    def _render_target_heatmap(self, heatmap, bboxes, labels, mask, min_overlap=0.3, alpha=0.54, method=None, eps=1e-8):
        """Render target heatmap for 1 image

        Args
            heatmap: tensor to render heatmap to
            bboxes: boxes in cxcwh format
            labels: labels of the boxes
            min_overlap: for cornernet method
            alpha: for ttfnet method
        """
        heatmap_height, heatmap_width = heatmap.shape[-2:]

        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        mask = mask.cpu().numpy()

        x_indices = (bboxes[...,0] * heatmap_width).astype(np.int32)
        y_indices = (bboxes[...,1] * heatmap_height).astype(np.int32)
        widths = bboxes[...,2] * heatmap_width
        heights = bboxes[...,3] * heatmap_height
        
        # for i in range(len(labels)):
        for x, y, w, h, label, m in zip(x_indices, y_indices, widths, heights, labels, mask):
            if m == 1:
                continue
            radius_w, radius_h = self._get_gaussian_radius(w, h, min_overlap=min_overlap, alpha=alpha, method=method)
            std_x = radius_w / 3
            std_y = radius_h / 3
            radius_w = int(radius_w)
            radius_h = int(radius_h)
            
            left   = np.minimum(x, radius_w)
            right  = np.minimum(heatmap_width - x, radius_w+1)
            top    = np.minimum(y, radius_h+1)
            bottom = np.minimum(heatmap_height - y, radius_h+1)

            grid_y = torch.arange(-radius_h, radius_h+1, device=heatmap.device).view(-1,1)
            grid_x = torch.arange(-radius_w, radius_w+1, device=heatmap.device).view(1,-1)

            gaussian = (-(grid_x.square() / (2*std_x*std_x+eps) + grid_y.square() / (2*std_y*std_y + eps))).exp()
            gaussian[gaussian < torch.finfo(gaussian.dtype).eps * torch.max(gaussian)] = 0

            masked_heatmap = heatmap[label, y-top:y+bottom, x-left:x+right]
            masked_gaussian = gaussian[radius_h-top:radius_h+bottom, radius_w-left:radius_w+right]
            torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    
        return heatmap

    def _get_gaussian_radius(self, width, height, min_overlap=0.3, alpha=0.54, method=None):
        """Get radius for the Gaussian kernel. From CornerNet https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
        """
        if method is None:
            method = self.target_method
        
        # ttfnet method
        if method == "ttfnet":
            return width/2 * alpha, height/2 * alpha

        # cornernet method
        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 * b1 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 * b2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 * b3 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        r = np.minimum(r1,r2)
        r = np.minimum(r,r3)
        r = np.maximum(r,0)
        return r, r

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

    def compute_loss(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], eps=1e-8):
        # overall steps
        # 1. calculate quantized (integer) box centers locations
        # 2. gather outputs along box center locations
        # 3. calculate predicted box coordinates
        # 4. apply loss function
        pred_box_map = pred["box_2d"]   # NCHW
        target_box = target["bboxes"]   # N x num_detections x 4, cxcywh format
        mask = target["mask"]           # N x num_detections
        
        batch_size, channels, output_height, output_width = pred_box_map.shape

        # 1. scale up to feature map size and convert to integer
        x_indices = target_box[...,0].clone() * output_width
        y_indices = target_box[...,1].clone() * output_height
        
        xy_indices = y_indices.long() * output_width + x_indices.long()
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size, channels, -1))

        # 2. gather outputs: left, top, right, bottom
        pred_box_map = pred_box_map.view(batch_size, channels, -1)          # flatten
        pred_box = torch.gather(pred_box_map, dim=-1, index=xy_indices)

        # 3. quantized xy (floor) aligned to center of the cell (+0.5)
        aligned_x = (torch.floor(x_indices) + 0.5)
        aligned_y = (torch.floor(y_indices) + 0.5)
        pred_box[:,0,:] = aligned_x - pred_box[:,0,:]   # x1 = x - left
        pred_box[:,1,:] = aligned_y - pred_box[:,1,:]   # y1 = y - top
        pred_box[:,2,:] = aligned_x + pred_box[:,2,:]   # x2 = x + right
        pred_box[:,3,:] = aligned_y + pred_box[:,3,:]   # y2 = y + bottom

        # 4. cxcywh to x1y1x2y2 and apply loss
        target_box = convert_cxcywh_to_x1y1x2y2(target_box, inplace=False).swapaxes(1,2)
        loss = self.loss_function(pred_box, target_box) * mask.unsqueeze(1)
        loss = loss.sum() / (mask.sum() + eps)
        return loss

class TimeDisplacementHead:
    pass

def build_output_heads(config: Union[str, Dict], in_channels):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["output_heads"]

    output_heads = nn.ModuleDict()
    output_head_mapper = {
        "heatmap": HeatmapHead,
        "box_2d": Box2DHead
    }
    for name, params in config.items():
        head = output_head_mapper[name](in_channels, **params)
        output_heads[name] = head
    
    return output_heads
