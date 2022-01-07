from typing import Dict, Union
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..losses import CornerNetFocalLossWithLogits, QualityFocalLossWithLogits
from ..losses import IoULoss, GIoULoss, DIoULoss, CIoULoss
from ..utils import convert_cxcywh_to_x1y1x2y2, load_config

class BaseHead(nn.Sequential):
    # Reference implementations
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L125    use num_filters = 256
    # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py#L5      use num_filters = in_channels
    
    def __init__(self, in_channels, out_channels, width=256, depth=1, init_bias=None):
        super().__init__()
        for i in range(depth):
            in_c = in_channels if i == 0 else width
            conv = nn.Conv2d(in_c, width, 3, padding=1)
            self.add_module(f"conv{i}", conv)
            self.add_module(f"relu{i}", nn.ReLU(inplace=True))
            
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

        self.out_conv = nn.Conv2d(width, out_channels, 1)
        if init_bias is not None:
            self.out_conv.bias.data.fill_(init_bias)
    
    def compute_loss(self, pred, target, eps=1e-8):
        raise NotImplementedError()

class HeatmapHead(BaseHead):
    _loss_mapper = {
        "cornernet_focal": CornerNetFocalLossWithLogits,
        "quality_focal": QualityFocalLossWithLogits
    }

    def __init__(self, in_channels, num_classes, width=256, depth=1, heatmap_prior=0.1, target_method="cornernet", loss_function="cornernet_focal", loss_weight=1):
        init_bias = math.log(heatmap_prior/(1-heatmap_prior))
        super().__init__(in_channels, num_classes, width=width, depth=depth, init_bias=init_bias)
        self.num_classes = num_classes
        self.target_method = target_method
        self.loss_function = self._loss_mapper[loss_function]()
        self.loss_weight = loss_weight

    def compute_loss(self, pred, target, eps=1e-8):
        heatmap = pred["heatmap"]
        bboxes = target["bboxes"]
        labels = target["labels"]
        mask = target["mask"]

        target_heatmap = self._render_target_heatmap(heatmap.shape, bboxes, labels, mask, device=heatmap.device)    # target heatmap
        loss = self.loss_function(heatmap, target_heatmap) / (mask.sum() + eps)                                     # apply focal loss

        return loss

    @classmethod
    def gather_topk(cls, heatmap: torch.Tensor, nms_kernel: int=3, num_detections: int=300):
        """Gather top k detections from heatmap
        """
        batch_size = heatmap.shape[0]

        # 1. pseudo-nms via max pool
        padding = (nms_kernel - 1) // 2
        nms_mask = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=padding) == heatmap
        heatmap = heatmap * nms_mask
        
        # 2. since box regression is shared, we only consider the best candidate at each heatmap location
        heatmap, labels = torch.max(heatmap, dim=1)

        # 3. flatten to run topk
        heatmap = heatmap.view(batch_size, -1)
        labels = labels.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(heatmap, num_detections)
        topk_labels = torch.gather(labels, dim=-1, index=topk_indices)

        return topk_scores, topk_indices, topk_labels

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

        # scale up to output heatmap dimensions
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

    def __init__(
        self,
        in_channels,
        loss_function="l1",
        loss_weight=1,
        log_box=False,
        box_multiplier=1,
        **kwargs
        ):
        super().__init__(in_channels, self.out_channels, **kwargs)
        self.loss_function = self._loss_mapper[loss_function](reduction="none")
        self.loss_weight = loss_weight
        self.log_box = log_box
        self.box_multiplier = box_multiplier

    def compute_loss(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], eps=1e-8):
        box_offsets = pred["box_2d"]        # N x 4 x out_h x out_w
        target_boxes = target["boxes"]      # N x num_detections x 4, cxcywh format
        mask = target["mask"]               # N x num_detections
        
        out_h, out_w = box_offsets.shape[-2:]

        # 1. scale target boxes to feature map size
        target_boxes = target_boxes.clone()
        target_boxes[...,[0,2]] *= out_w
        target_boxes[...,[1,3]] *= out_h

        # 2. get training samples. only center
        # TODO: 3x3 square
        indices = target_boxes[...,1].round() * out_w + target_boxes[...,0].round()
        pred_boxes = self.gather_and_decode(box_offsets, indices)
        
        # 3. convert to xyxy and apply loss
        target_boxes = convert_cxcywh_to_x1y1x2y2(target_boxes)
        pred_box = pred_boxes.swapaxes(1,2)
        loss = self.loss_function(pred_box, target_boxes) * mask.unsqueeze(-1)
        loss = loss.sum() / (mask.sum() + eps)
        return loss

    def gather_and_decode(self, box_offsets: torch.Tensor, indices: torch.Tensor, normalize_boxes: bool=False):
        """Gather 2D bounding boxes at given indices
        """
        batch_size, _, out_h, out_w = box_offsets.shape

        cx = indices % out_w + 0.5
        cy = indices // out_w + 0.5

        box_offsets = box_offsets.view(batch_size, 4, -1)
        boxes = torch.zeros((batch_size, len(indices), 4))

        if self.log_box:
            box_offsets = torch.exp(box_offsets)
        if self.box_multiplier > 1:
            box_offsets *= self.box_multiplier
        box_offsets = box_offsets.clamp_min(0)

        boxes[...,0] = cx - torch.gather(box_offsets[:,0], dim=-1, index=indices)   # x1 = cx - left
        boxes[...,1] = cy - torch.gather(box_offsets[:,1], dim=-1, index=indices)   # y1 = cy - top
        boxes[...,2] = cx + torch.gather(box_offsets[:,2], dim=-1, index=indices)   # x2 = cx + right
        boxes[...,3] = cy + torch.gather(box_offsets[:,3], dim=-1, index=indices)   # y2 = cy + bottom

        if normalize_boxes:      # convert to normalized coordinates
            boxes[...,[0,2]] /= out_w
            boxes[...,[1,3]] /= out_h
        
        return boxes

class EmbeddingHead(BaseHead):
    """FairMOT head. Paper: https://arxiv.org/abs/2004.01888
    """
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L3322
    # hidden channels = 64 as recommended by FairMOT
    _loss_mapper = {
        "ce": nn.CrossEntropyLoss
    }

    def __init__(self, in_channels, max_track_ids=1000, emb_dim=64, width=256, depth=1, init_bias=None, loss_function="ce", loss_weight=1):
        super().__init__(in_channels, emb_dim, width=width, depth=depth, init_bias=init_bias)
        self.loss_function = self._loss_mapper[loss_function](reduction="none")
        self.reid_dim = emb_dim
        self.loss_weight = loss_weight
        
        # used during training only
        self.classifier = nn.Sequential(
            nn.Linear(self.reid_dim, self.reid_dim, bias=False),
            nn.BatchNorm1d(self.reid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reid_dim, max_track_ids)
        )

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

    @classmethod
    def gather_at_indices(cls, reid: torch.Tensor, indices: torch.Tensor):
        """Gather ReID embeddings at given indices
        """
        batch_size, embedding_size, _, _ = reid.shape

        reid = reid.view(batch_size, embedding_size, -1)
        indices = indices.unsqueeze(1).expand(batch_size, embedding_size, -1)
        embeddings = torch.gather(reid, dim=-1, index=indices)
        embeddings = embeddings.swapaxes(1,2)
        return embeddings


def build_output_heads(config: Union[str, Dict], in_channels):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["output_heads"]

    output_heads = nn.ModuleDict()
    output_head_mapper = {
        "heatmap": HeatmapHead,
        "box_2d": Box2DHead,
        "reid": EmbeddingHead
    }
    for name, params in config.items():
        head = output_head_mapper[name](in_channels, **params)
        output_heads[name] = head
    
    return output_heads
