from copy import deepcopy
from typing import Any, Callable, List, Dict, Tuple, Union
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import box_convert
import albumentations as A
from albumentations.pytorch import ToTensorV2
from vision_toolbox import backbones, necks
from vision_toolbox.components import ConvBnAct

from .meta import GenericHead, GenericLightning
from ..losses import heatmap_losses, box_losses
from ..datasets.coco import CocoDetection, collate_fn
from ..datasets import transforms
from ..eval.coco import CocoEvaluator


_transforms = {**A.__dict__, **transforms.__dict__}


class _FixedRadius:
    def __init__(self, r: float=1.):
        self.r = r

    def __call__(self, w, h):
        return self.r, self.r
    
class _TTFNetRadius:
    def __init__(self, alpha: float=0.54):
        self.alpha = alpha
    
    def __call__(self, w, h):
        return w/2 * self.alpha, h/2 * self.alpha

class _CornerNetRadius:
    def __init__(self, min_overlap: float=0.3):
        self.min_overlap = min_overlap
    
    # Explanation: https://github.com/princeton-vl/CornerNet/issues/110
    # Source: https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    def __call__(self, w, h):
        a1 = 1
        b1 = h + w
        c1 = w * h * (1 - self.min_overlap) / (1 + self.min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (h + w)
        c2 = (1 - self.min_overlap) * w * h
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * self.min_overlap
        b3 = -2 * self.min_overlap * (h + w)
        c3 = (self.min_overlap - 1) * w * h
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        r = min(r1, r2, r3)
        return r, r


_heatmap_targets = {
    "fixed": _FixedRadius,
    "ttfnet": _TTFNetRadius,
    "cornernet": _CornerNetRadius
}


class CenterNet(GenericLightning):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        pretrained_backbone: bool=False,
        neck: str="FPN",
        neck_config: Dict[str, Any]=None,

        # head configuration
        head_width: int=256,
        head_depth: int=3,
        head_block: Callable=ConvBnAct,
        heatmap_prior: float=0.01,
        box_init_bias: float=None,

        # box params
        box_log: bool=False,
        box_multiplier: float=1.,
        box_loss: str="L1Loss",

        # heatmap params
        heatmap_loss: str="CornerNetFocalLoss",
        heatmap_target: str="cornernet",
        heatmap_target_params: Dict[str, float]=None,
        
        # inference config
        nms_kernel: int=3,
        num_detections: int=100,

        # data
        batch_size: int=8,
        num_workers: int=2,
        train_data: Dict[str, Any]=None,
        val_data: Dict[str, Any]=None,

        **kwargs
    ):
        self.save_hyperparameters()
        if neck_config is None:
            neck_config = {}
        backbone: backbones.BaseBackbone = backbones.__dict__[backbone](pretrained=pretrained_backbone)
        neck: necks.BaseNeck = necks.__dict__[neck](backbone.get_out_channels(), **neck_config)

        head_in_c = neck.get_out_channels()
        heatmap_init_bias = math.log(heatmap_prior/(1-heatmap_prior))
        heads = nn.Module()
        heads.add_module("heatmap", GenericHead(head_in_c, num_classes, width=head_width, depth=head_depth, block=head_block, init_bias=heatmap_init_bias))
        heads.add_module("box_2d", GenericHead(head_in_c, 4, width=head_width, depth=head_depth, block=head_block, init_bias=box_init_bias))

        super().__init__(backbone, neck, heads, **kwargs)
        self.num_classes = num_classes
        self.stride = backbone.stride // neck.stride
        self.evaluator = CocoEvaluator(num_classes)

        self.heatmap_loss = heatmap_losses.__dict__[heatmap_loss]()
        if heatmap_target_params is None:
            heatmap_target_params = {}
        self.heatmap_radius = _heatmap_targets[heatmap_target](**heatmap_target_params)
        
        self.box_loss = box_losses.__dict__[box_loss](reduction='sum')
        self.box_log = box_log
        self.box_multiplier = box_multiplier

        self.nms_kernel = nms_kernel
        self.num_detections = num_detections
    
    def compute_loss(self, outputs, targets):
        heatmap = outputs["heatmap"]
        box_offsets = outputs["box_2d"]
        out_h, out_w = heatmap.shape[-2:]

        num_dets = 0
        num_boxes = 0
        target_heatmap = torch.zeros_like(heatmap)
        box_loss = torch.tensor(0., dtype=heatmap.dtype, device=heatmap.dtype)
        for i, instances in enumerate(targets):
            if len(instances["labels"]) == 0:
                continue
            
            # heatmap
            self.update_heatmap(target_heatmap[i,...], instances['boxes'], instances['labels'])
            num_dets += len(instances['labels'])

            # box
            boxes = []
            indices = []
            for box in instances["boxes"]:
                x, y, w, h = [d/self.stride for d in box]       # convert to feature map coordinates
                cx, cy = int(x + w/2), int(y + h/2)
                cxs = [d for d in [cx-1, cx, cx+1] if d > 0 and d < out_w-1]
                cys = [d for d in [cy-1, cy, cy+1] if d > 0 and d < out_h-1]

                for cx in cxs:
                    for cy in cys:
                        boxes.append(box)       # xywh
                        indices.append(cy * out_w + cx)

            target_boxes = box_convert(torch.tensor(boxes), "xywh", "xyxy").to(box_offsets.device)
            pred_boxes = self.gather_and_decode_boxes(box_offsets[i], torch.tensor(indices).to(box_offsets.device))
            box_loss += self.box_loss(pred_boxes, target_boxes)
            num_boxes += len(boxes)

            # # 1. convert target boxes to xyxy and get center points
            # img_boxes = box_convert(torch.tensor(instances["boxes"]), "xywh", "xyxy")
            # cx = (img_boxes[...,0] + img_boxes[...,2]) / 2 / self.stride        # center points in output feature map coordinates
            # cy = (img_boxes[...,1] + img_boxes[...,3]) / 2 / self.stride

            # # 2. gather training samples. only center point
            # # TODO: 3x3 square
            # # method: permutate, get (cx-1, cx, cx+1) x (cy-1, cy, cy+1)
            # indices = cy.long() * out_w + cx.long()
            # pred_boxes = Box2DHead.gather_and_decode_boxes(box_offsets[i], indices.to(box_offsets.device), stride=self.stride, **self.box_params)

            # # 3. apply loss
            # loss += self.loss_function(pred_boxes, img_boxes.to(box_offsets.device))
            # num_dets += len(instances["boxes"])

        heatmap_loss = self.heatmap_loss(heatmap, target_heatmap) / max(1, num_dets)
        box_loss = box_loss / max(1, num_boxes)

        return {
            "heatmap": heatmap_loss,
            "box_2d": box_loss,
            "total": heatmap_loss + box_loss
        }


    # TODO: make this torchscript-able
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/coco.py
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    def update_heatmap(self, heatmap: torch.Tensor, boxes: Tuple[Tuple[int]], labels: Tuple[int]):
        """Render target heatmap for a batch of images
        """
        out_h, out_w = heatmap.shape[-2:]
        for box, label in zip(boxes, labels):
            # scale up to heatmap dimensions
            x, y, w, h = [i / self.stride for i in box]
            cx = round(x + w/2)
            cy = round(y + h/2)
            
            r_x, r_y = self.heatmap_radius(w, h)
            # TODO: check CenterNet, mmdet implementation, and CenterNet2
            r_x = max(0, round(r_x))
            r_y = max(0, round(r_y))
            std_x = r_x / 3 + 1/6
            std_y = r_y / 3 + 1/6
            
            l = min(cx, r_x)
            t = min(cy, r_y)
            r = min(out_w - cx, r_x+1)
            b = min(out_h - cy, r_y+1)

            # only gaussian and heatmap are on gpu
            grid_y = torch.arange(-r_y, r_y+1, device=heatmap.device).view(-1,1)
            grid_x = torch.arange(-r_x, r_x+1, device=heatmap.device).view(1,-1)

            gaussian = grid_x.square() / (2 * std_x * std_x) + grid_y.square() / (2 * std_y * std_y)
            gaussian = torch.exp(-gaussian)
            gaussian[gaussian < torch.finfo(gaussian.dtype).eps * torch.max(gaussian)] = 0

            masked_heatmap = heatmap[label, cy-t:cy+b, cx-l:cx+r]
            masked_gaussian = gaussian[r_y-t:r_y+b, r_x-l:r_x+r]
            torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        preds = self.decode_detections(outputs['heatmap'], outputs['box_2d'])
        
        preds["boxes"] = box_convert(preds["boxes"], "xyxy", "xywh")                    # coco box format
        preds = {k: v.cpu().numpy() for k, v in preds.items()}                          # convert to numpy
        preds = [{k: v[i] for k, v in preds.items()} for i in range(images.shape[0])]   # convert to list of images
        
        targets = [{k: np.array(target[k]) for k in ("boxes", "labels")} for target in targets]     # filter keys and convert to numpy array
        
        self.evaluator.update(preds, targets)

    def validation_epoch_end(self, outputs):
        metrics = self.evaluator.get_metrics()
        self.evaluator.reset()

        for k, v in metrics.items():
            self.log(f"val/{k}", v)
 
    def get_dataloader(self, train=True):
        config = self.hparams.train_data if train else self.hparams.val_data

        ts = []
        for t in config["transforms"]:
            t_fn = _transforms[t["name"]]
            init_args = t["init_args"] if "init_args" in t else {}
            ts.append(t_fn(**init_args))
        ts.append(ToTensorV2())
        
        transforms = A.Compose(ts, bbox_params=dict(format="coco", label_fields=["labels"], min_area=1))
        ds = CocoDetection(config["img_dir"], config["ann_json"], transforms=transforms)

        return DataLoader(
            ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
            shuffle=train, collate_fn=collate_fn, pin_memory=True
        )

    def decode_detections(self, heatmap: torch.Tensor, box_offsets: torch.Tensor, normalize_boxes: bool=False) -> Dict[str, torch.Tensor]:
        """Decode model outputs for detection task

        Args:
            heatmap: heatmap output
            box_offsets: box_2d output
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
        
        Returns: a Dict with keys boxes, scores, labels
        """
        scores, indices, labels = self.get_topk_from_heatmap(heatmap, k=self.num_detections)
        boxes = self.gather_and_decode_boxes(box_offsets, indices, normalize_boxes=normalize_boxes)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }

    def get_topk_from_heatmap(self, heatmap: torch.Tensor):
        """Gather top k detections from heatmap

        Args:
            heatmap: (N, num_classes, H, W) or (num_classes, H, W)
        
        Returns:
            scores, indices, labels: (N, k) or (k,)
        """
        batch_size = heatmap.shape[0]

        # 1. pseudo-nms via max pool
        padding = (self.nms_kernel - 1) // 2
        nms_mask = F.max_pool2d(heatmap, kernel_size=self.nms_kernel, stride=1, padding=padding) == heatmap
        heatmap = heatmap * nms_mask
        
        # 2. since box regression is shared, we only consider the best candidate at each heatmap location
        heatmap, labels = torch.max(heatmap, dim=1)

        # 3. flatten and get topk
        heatmap = heatmap.view(batch_size, -1)
        labels = labels.view(batch_size, -1)
        scores, indices = torch.topk(heatmap, self.num_detections)
        labels = torch.gather(labels, dim=-1, index=indices)
    
        return scores, indices, labels

    def gather_and_decode_boxes(self, box_offsets: torch.Tensor, indices: torch.Tensor, normalize_boxes: bool=False) -> torch.Tensor:
        """Gather 2D bounding boxes at given indices

        Args:
            box_offsets: (N, 4, H, W) or (4, H, W)
            indices: (N, num_dets) or (num_dets,)

        Returns:
            boxes: (N, num_dets, 4) or (num_dets, 4)
        """
        out_h, out_w = box_offsets.shape[-2:]
        cx = indices % out_w + 0.5
        cy = indices // out_w + 0.5

        # decoded = multiplier x exp(encoded)
        box_offsets = box_offsets.flatten(start_dim=-2)
        if self.box_log:
            box_offsets = torch.exp(box_offsets)
        box_offsets = box_offsets * self.box_multiplier     # *= will multiply inplace -> cannot call .backward()
        box_offsets = box_offsets.clamp_min(0)

        # boxes are in output feature maps coordinates
        x1 = cx - torch.gather(box_offsets[...,0,:,:], dim=-1, index=indices)       # x1 = cx - left
        y1 = cy - torch.gather(box_offsets[...,1,:,:], dim=-1, index=indices)       # y1 = cy - top
        x2 = cx + torch.gather(box_offsets[...,2,:,:], dim=-1, index=indices)       # x2 = cx + right
        y2 = cy + torch.gather(box_offsets[...,3,:,:], dim=-1, index=indices)       # y2 = cy + bottom
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        if normalize_boxes:             # convert to normalized coordinates
            boxes[...,[0,2]] /= out_w
            boxes[...,[1,3]] /= out_h
        else:
            boxes *= self.stride        # convert to input coordinates

        return boxes
