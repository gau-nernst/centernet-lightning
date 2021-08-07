from typing import Dict, Union

import numpy as np
import torch
from torch import nn

from ..losses import CornerNetFocalLossWithLogits, QualityFocalLossWithLogits
from ..losses import IoULoss, GIoULoss, DIoULoss, CIoULoss
from ..utils import convert_cxcywh_to_x1y1x2y2, load_config

class BaseHead(nn.Module):
    # Reference implementations
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L125    use num_filters = 256
    # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py#L5      use num_filters = in_channels
    
    def __init__(self, in_channels, out_channels, hidden_channels=256, init_bias=None):
        super().__init__()
        conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(hidden_channels, out_channels, 1)
        
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        if init_bias is not None:
            conv2.bias.data.fill_(init_bias)
        
        self.head = nn.Sequential(conv1, relu, conv2)

    def forward(self, x):
        return self.head(x)
    
    def compute_loss(self, pred, target, eps=1e-8):
        raise NotImplementedError()

class HeatmapHead(BaseHead):
    _loss_mapper = {
        "cornernet_focal": CornerNetFocalLossWithLogits,
        "quality_focal": QualityFocalLossWithLogits
    }

    def __init__(self, in_channels, num_classes, hidden_channels=256, init_bias=-2.19, target_method="cornernet", loss_function="cornernet_focal", loss_weight=1):
        super().__init__(in_channels, num_classes, hidden_channels=hidden_channels, init_bias=init_bias)
        self.num_classes = num_classes
        self.target_method = target_method
        self.loss_function = self._loss_mapper[loss_function]()
        self.loss_weight = loss_weight

    def compute_loss(self, pred, target, eps=1e-8):
        # overall steps
        # 1. calculate target heatmap
        # 2. apply loss
        heatmap = pred["heatmap"]
        bboxes = target["bboxes"]
        labels = target["labels"]
        mask = target["mask"]

        target_heatmap = self._render_target_heatmap(heatmap.shape, bboxes, labels, mask, device=heatmap.device)

        loss = self.loss_function(heatmap, target_heatmap) / (mask.sum() + eps)

        return loss

    def _render_target_heatmap(self, heatmap_shape, bboxes, labels, mask, min_overlap=0.3, alpha=0.54, method=None, device="cpu", eps=1e-8):
        """Render target heatmap for a batch of images

        Args
            heatmap_shape: shape of heatmap
            bboxes: boxes in cxcwh format
            labels: labels of the boxes
            min_overlap: for cornernet method
            alpha: for ttfnet method
        """
        batch_size, _, heatmap_height, heatmap_width = heatmap_shape
        heatmap = torch.zeros(heatmap_shape, device=device)

        # doing iteration on cpu is faster
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        mask = mask.cpu().numpy()

        x_indices = (bboxes[...,0] * heatmap_width).astype(np.int32)
        y_indices = (bboxes[...,1] * heatmap_height).astype(np.int32)
        widths = bboxes[...,2] * heatmap_width
        heights = bboxes[...,3] * heatmap_height
        
        for b in range(batch_size):
            for x, y, w, h, label, m in zip(x_indices[b], y_indices[b], widths[b], heights[b], labels[b], mask[b]):
                if m == 0:
                    continue
                radius_w, radius_h = self._get_gaussian_radius(w, h, min_overlap=min_overlap, alpha=alpha, method=method)
                std_x = radius_w / 3
                std_y = radius_h / 3
                radius_w = radius_w.astype(np.int32)
                radius_h = radius_h.astype(np.int32)
                
                left   = np.minimum(x, radius_w)
                right  = np.minimum(heatmap_width - x, radius_w+1)
                top    = np.minimum(y, radius_h)
                bottom = np.minimum(heatmap_height - y, radius_h+1)

                # only gaussian and heatmap are on gpu
                grid_y = torch.arange(-radius_h, radius_h+1, device=heatmap.device).view(-1,1)
                grid_x = torch.arange(-radius_w, radius_w+1, device=heatmap.device).view(1,-1)

                gaussian = (-(grid_x.square() / (2*std_x*std_x+eps) + grid_y.square() / (2*std_y*std_y + eps))).exp()
                gaussian[gaussian < torch.finfo(gaussian.dtype).eps * torch.max(gaussian)] = 0

                masked_heatmap = heatmap[b, label, y-top:y+bottom, x-left:x+right]
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

class Box2DHead(BaseHead):
    _loss_mapper = {
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "iou": IoULoss,
        "giou": GIoULoss,
        "diou": DIoULoss,
        "ciou": CIoULoss
    }
    out_channels = 4

    def __init__(self, in_channels, hidden_channels=256, init_bias=None, loss_function="l1", loss_weight=1):
        super().__init__(in_channels, self.out_channels, hidden_channels=hidden_channels, init_bias=init_bias)
        self.loss_function = self._loss_mapper[loss_function](reduction="none")
        self.loss_weight = loss_weight

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
        target_box = target_box.clone()
        target_box[...,[0,2]] *= output_width
        target_box[...,[1,3]] *= output_height

        x_indices = target_box[...,0].long()
        y_indices = target_box[...,1].long()
        xy_indices = y_indices * output_width + x_indices
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size, channels, -1))

        # 2. gather outputs: left, top, right, bottom
        pred_box_map = torch.clamp_min(pred_box_map, 0)
        pred_box_map = pred_box_map.view(batch_size, channels, -1)          # flatten
        pred_box = torch.gather(pred_box_map, dim=-1, index=xy_indices)

        # 3. quantized xy (floor) aligned to center of the cell (+0.5)
        aligned_x = torch.floor(target_box[...,0]) + 0.5
        aligned_y = torch.floor(target_box[...,1]) + 0.5
        pred_box[:,0,:] = aligned_x - pred_box[:,0,:]   # x1 = x - left
        pred_box[:,1,:] = aligned_y - pred_box[:,1,:]   # y1 = y - top
        pred_box[:,2,:] = aligned_x + pred_box[:,2,:]   # x2 = x + right
        pred_box[:,3,:] = aligned_y + pred_box[:,3,:]   # y2 = y + bottom

        # 4. cxcywh to x1y1x2y2 and apply loss
        target_box = convert_cxcywh_to_x1y1x2y2(target_box, inplace=False)
        pred_box = pred_box.swapaxes(1,2)
        loss = self.loss_function(pred_box, target_box) * mask.unsqueeze(-1)
        loss = loss.sum() / (mask.sum() + eps)
        return loss

class ReIDHead(BaseHead):
    """FairMOT head. Paper: https://arxiv.org/abs/2004.01888
    """
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L3322
    # hidden channels = 64 as recommended by FairMOT
    _loss_mapper = {
        "ce": nn.CrossEntropyLoss
    }

    def __init__(self, in_channels, max_track_ids=1000, reid_dim=64, hidden_channels=256, init_bias=None, loss_function="ce", loss_weight=1):
        super().__init__(in_channels, reid_dim, hidden_channels=hidden_channels, init_bias=init_bias)
        self.loss_function = self._loss_mapper[loss_function](reduction="none")
        self.reid_dim = reid_dim
        self.loss_weight = loss_weight
        
        # used during training only
        self.classifier = self._make_classification_head(max_track_ids)

    def compute_loss(self, pred, target, eps=1e-8):
        reid_embeddings = pred["reid"]
        target_box = target["bboxes"]
        track_ids = target["ids"]
        mask = target["mask"]

        batch_size, channels, output_height, output_width = reid_embeddings.shape

        # scale up to feature map size and convert to integer
        target_box = target_box.clone()
        target_box[...,[0,2]] *= output_width
        target_box[...,[1,3]] *= output_height

        x_indices = target_box[...,0].long()
        y_indices = target_box[...,1].long()
        xy_indices = y_indices * output_width + x_indices
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size, channels, -1))

        reid_embeddings = reid_embeddings.view(batch_size, channels, -1)
        reid_embeddings = torch.gather(reid_embeddings, dim=-1, index=xy_indices)
        
        # flatten, pass through classifier, apply cross entropy loss
        reid_embeddings = reid_embeddings.swapaxes(1, 2).reshape(-1, channels)  # N x C x num_detections -> (N x num_detections) x C
        logits = self.classifier(reid_embeddings)
        loss = self.loss_function(logits, track_ids.view(-1).long()) * mask.view(-1)
        loss = loss.sum() / (mask.sum() + eps)

        return loss

    def _make_classification_head(self, num_classes):
        # 2-layer MLP
        head = nn.Sequential(
            nn.Linear(self.reid_dim, self.reid_dim, bias=False),
            nn.BatchNorm1d(self.reid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reid_dim, num_classes)
        )
        return head

def build_output_heads(config: Union[str, Dict], in_channels):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["output_heads"]

    output_heads = nn.ModuleDict()
    output_head_mapper = {
        "heatmap": HeatmapHead,
        "box_2d": Box2DHead,
        "reid": ReIDHead
    }
    for name, params in config.items():
        head = output_head_mapper[name](in_channels, **params)
        output_heads[name] = head
    
    return output_heads
