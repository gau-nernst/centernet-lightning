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

from .meta import GenericHead, MetaCenterNet
from ..losses import heatmap_losses, box_losses
from ..datasets.coco import CocoDetection, collate_fn
from ..datasets import transforms
from ..eval.coco import CocoEvaluator


_transforms = {**A.__dict__, **transforms.__dict__}

class HeatmapHead(BaseHead):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stride: int=4,
        heatmap_prior: float=0.1,

        loss_function: str="CornerNetFocalLoss",
        radius_method: str="cornernet",
        radius: float=1,                # for fixed radius
        min_overlap: float=0.3,         # for cornetnet
        alpha: float=0.54,              # for ttfnet

        **base_head_kwargs
    ):
        radius_lookup = {
            "fixed": self.fixed_radius,
            "cornernet": self.cornernet_radius,
            "ttfnet": self.ttfnet_radius
        }
        assert radius_method in radius_lookup
        init_bias = math.log(heatmap_prior/(1-heatmap_prior))
        super().__init__(in_channels, num_classes, init_bias=init_bias, **base_head_kwargs)
        
        self.stride = stride
        self.loss_function = heatmap_losses.__dict__[loss_function]()

        self.radius = radius
        self.min_overlap = min_overlap
        self.alpha = alpha
        self.radius_fn = radius_lookup[radius_method]

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Union[List, int]]]) -> torch.Tensor:
        heatmap = outputs["heatmap"]

        target_heatmap = torch.zeros_like(heatmap)
        num_dets = 0
        for i, instances in enumerate(targets):
            self.update_heatmap(target_heatmap[i,...], instances["boxes"], instances["labels"])
            num_dets += len(instances["labels"])

        loss = self.loss_function(heatmap, target_heatmap) / max(1, num_dets)
        return loss

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
            
            r_x, r_y = self.radius_fn(w, h)
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

    def fixed_radius(self, w, h):
        return self.radius, self.radius
    
    # Explanation: https://github.com/princeton-vl/CornerNet/issues/110
    # Source: https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    def cornernet_radius(self, w, h):
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

    def ttfnet_radius(self, w, h):
        return w/2 * self.alpha, h/2 * self.alpha


class Box2DHead(BaseHead):
    def __init__(
        self,
        in_channels: int,
        stride: int=4,
        loss_function: str="L1Loss",

        # box parameterization        
        log_box: bool=False,
        box_multiplier: float=1.,
        
        **base_head_kwargs
    ):
        out_channels = 4
        super().__init__(in_channels, out_channels, **base_head_kwargs)
        self.stride = stride
        self.loss_function = box_losses.__dict__[loss_function](reduction="sum")
        self.box_params = {"log_box": log_box, "box_multiplier": box_multiplier}

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Tuple[Dict[str, Union[Tuple, int]]]) -> torch.Tensor:
        box_offsets = outputs["box_2d"]           # (N, 4, H, W)
        out_h, out_w = box_offsets.shape[-2:]

        num_dets = 0
        loss = torch.tensor(0., dtype=box_offsets.dtype, device=box_offsets.device)
        for i, instances in enumerate(targets):
            if len(instances["boxes"]) == 0:
                continue        # skip image without boxes

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
            pred_boxes = Box2DHead.gather_and_decode_boxes(box_offsets[i], torch.tensor(indices).to(box_offsets.device), stride=self.stride, **self.box_params)
            loss += self.loss_function(pred_boxes, target_boxes)
            num_dets += len(boxes)


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

        loss = loss / max(1, num_dets)
        return loss

    @staticmethod
    def gather_and_decode_boxes(
        box_offsets: torch.Tensor, indices: torch.Tensor, normalize_boxes: bool=False,
        log_box: bool=False, box_multiplier: float=1., stride: int=4
    ) -> torch.Tensor:
        """Gather 2D bounding boxes at given indices

        Args:
            box_offsets: (N, 4, H, W) or (4, H, W)
            indices: (N, num_dets) or (num_dets,)

        Returns:
            boxes: (N, num_dets, 4) or (num_dets, 4)
        """
        out_h, out_w = box_offsets.shape[-2:]
        squeeze = False
        if len(box_offsets.shape) == 3:             # add batch dim
            box_offsets = box_offsets.unsqueeze(0)
            indices = indices.unsqueeze(0)
            squeeze = True

        cx = indices % out_w + 0.5
        cy = indices // out_w + 0.5

        # decoded = multiplier x exp(encoded)
        box_offsets = box_offsets.flatten(start_dim=-2)
        if log_box:
            box_offsets = torch.exp(box_offsets)
        box_offsets = box_offsets * box_multiplier      # *= will multiply inplace -> cannot call .backward()
        box_offsets = box_offsets.clamp_min(0)

        # boxes are in output feature maps coordinates
        x1 = cx - torch.gather(box_offsets[:,0], dim=-1, index=indices)     # x1 = cx - left
        y1 = cy - torch.gather(box_offsets[:,1], dim=-1, index=indices)     # y1 = cy - top
        x2 = cx + torch.gather(box_offsets[:,2], dim=-1, index=indices)     # x2 = cx + right
        y2 = cy + torch.gather(box_offsets[:,3], dim=-1, index=indices)     # y2 = cy + bottom
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        if normalize_boxes:             # convert to normalized coordinates
            boxes[...,[0,2]] /= out_w
            boxes[...,[1,3]] /= out_h
        else:
            boxes *= stride        # convert to input coordinates

        if squeeze:
            boxes = boxes.squeeze(0)

        return boxes


class CenterNet(MetaCenterNet):
    def __init__(
        self,
        num_classes: int,
        backbone: str,
        pretrained_backbone: bool=False,
        neck: str="FPN",
        neck_config: Dict[str, Any]=None,
        heatmap_config: Dict[str, Any]=None,
        box2d_config: Dict[str, Any]=None,

        head_width: int=256,
        head_depth: int=3,
        head_block: Callable=ConvBnAct,
        heatmap_prior: float=0.01,
        box_init_bias: float=None,

        # inference config
        nms_kernel: int=3,
        num_detections: int=300,

        # data
        batch_size: int=8,
        num_workers: int=2,
        train_data: Dict[str, Any]=None,
        val_data: Dict[str, Any]=None,

        **kwargs
    ):
        neck_config = deepcopy(neck_config) if neck_config is not None else {}
        heatmap_config = deepcopy(heatmap_config) if heatmap_config is not None else {}
        box2d_config = deepcopy(box2d_config) if box2d_config is not None else {}

        backbone: backbones.BaseBackbone = backbones.__dict__[backbone](pretrained=pretrained_backbone)
        neck: necks.BaseNeck = necks.__dict__[neck](backbone.get_out_channels(), **neck_config)
        stride = backbone.stride // neck.stride

        # heatmap_head = HeatmapHead(neck.get_out_channels(), num_classes, stride=stride, **heatmap_config)
        # box2d_head = Box2DHead(neck.get_out_channels(), stride=stride, **box2d_config)
        # heads = {"heatmap": heatmap_head, "box_2d": box2d_head}

        head_in_c = neck.get_out_channels()
        heatmap_init_bias = math.log(heatmap_prior/(1-heatmap_prior))
        heads = nn.Module()
        heads.add_module("heatmap", GenericHead(head_in_c, num_classes, width=head_width, depth=head_depth, block=head_block, init_bias=heatmap_init_bias))
        heads.add_module("box_2d", GenericHead(head_in_c, 4, width=head_width, depth=head_depth, block=head_block, init_bias=box_init_bias))

        super().__init__(backbone, neck, heads, **kwargs)
        self.num_classes = num_classes
        self.stride = stride
        self.evaluator = CocoEvaluator(num_classes)
        # self.decode_params = {
        #     "nms_kernel": nms_kernel,
        #     "num_detections": num_detections,
        #     "stride": stride,
        #     **box2d_head.box_params
        # }
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor):
        outputs = self.get_output_dict(x)
        return outputs["heatmap"], outputs["box_2d"]

    def predict(self, x: torch.Tensor, **kwargs):
        """Override decode parameters with keyword arguments. Decode paramms: nms_kernel, num_detections
        """
        decode_params = deepcopy(self.decode_params)
        for k, v in kwargs.items():
            if k in decode_params:
                decode_params[k] = v

        heatmap, box_offsets = self(x)
        return CenterNet.decode_detections(heatmap.sigmoid(), box_offsets, **decode_params)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        preds = self.predict(images, num_detections=100)                                # pycocotools consider max of 100 detections
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

    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)

    @staticmethod
    def decode_detections(
        heatmap: torch.Tensor, box_offsets: torch.Tensor, normalize_boxes: bool=False,
        num_detections: int=300, nms_kernel: int=3,
        log_box: bool=False, box_multiplier: float=1., stride: int=4
    ) -> Dict[str, torch.Tensor]:
        """Decode model outputs for detection task

        Args:
            heatmap: heatmap output
            box_offsets: box_2d output
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
            img_widths
            img_heights
        
        Returns: a Dict with keys boxes, scores, labels
        """
        scores, indices, labels = CenterNet.get_topk_from_heatmap(heatmap, k=num_detections, nms_kernel=nms_kernel)
        boxes = Box2DHead.gather_and_decode_boxes(
            box_offsets, indices, normalize_boxes=normalize_boxes, 
            log_box=log_box, box_multiplier=box_multiplier, stride=stride
        )
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }

    @staticmethod
    def get_topk_from_heatmap(heatmap: torch.Tensor, k: int=300, nms_kernel: int=3):
        """Gather top k detections from heatmap

        Args:
            heatmap: (N, num_classes, H, W) or (num_classes, H, W)
        
        Returns:
            scores, indices, labels: (N, k) or (k,)
        """
        squeeze = False
        if len(heatmap.shape) == 3:         # add batch dim
            heatmap = heatmap.unsqueeze(0)
            squeeze = True
        batch_size = heatmap.shape[0]

        # 1. pseudo-nms via max pool
        padding = (nms_kernel - 1) // 2
        nms_mask = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=padding) == heatmap
        heatmap = heatmap * nms_mask
        
        # 2. since box regression is shared, we only consider the best candidate at each heatmap location
        heatmap, labels = torch.max(heatmap, dim=1)

        # 3. flatten and get topk
        heatmap = heatmap.view(batch_size, -1)
        labels = labels.view(batch_size, -1)
        scores, indices = torch.topk(heatmap, k)
        labels = torch.gather(labels, dim=-1, index=indices)

        if squeeze:
            scores = scores.squeeze(0)
            indices = indices.squeeze(0)
            labels = labels.squeeze(0)

        return scores, indices, labels
