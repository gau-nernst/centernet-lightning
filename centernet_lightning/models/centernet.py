from typing import Any, Callable, Dict, Tuple
import math
import itertools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import box_convert

from vision_toolbox import backbones, necks
from vision_toolbox.components import ConvBnAct

from .meta import GenericHead, GenericLightning
from ..losses import heatmap_losses, box_losses
from ..datasets.coco import CocoDetection, coco_detection_collate_fn, parse_albumentations_transforms
from ..eval.coco import CocoEvaluator


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
        box_loss: str="L1Loss",
        box_loss_weight: float=0.1,
        box_log: bool=False,
        box_multiplier: float=1.,
        
        # heatmap params
        heatmap_loss: str="CornerNetFocalLoss",
        heatmap_loss_weight: float=1.,
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
        self.box_loss = box_losses.__dict__[box_loss](reduction='sum')
        if heatmap_target_params is None:
            heatmap_target_params = {}
        self.heatmap_radius = _heatmap_targets[heatmap_target](**heatmap_target_params)
        
    def compute_loss(self, outputs, targets):
        heatmap = outputs["heatmap"]
        box_offsets = outputs["box_2d"]
        out_h, out_w = heatmap.shape[-2:]
        dtype = heatmap.dtype
        device = self.device

        num_dets = 0
        num_boxes = 0
        target_heatmap = torch.zeros_like(heatmap)
        box_loss = torch.tensor(0., dtype=dtype, device=device)
        for i, instances in enumerate(targets):
            if len(instances["labels"]) == 0:
                continue

            boxes = np.array(instances['boxes']) / self.stride  # convert to feature map coordinates
            centers = boxes[...,:2] + boxes[...,2:] / 2         # cx = x + w/2
            centers = centers.round().astype(int)

            # heatmap
            radii = [self.heatmap_radius(w, h) for w, h in boxes[...,2:]]
            self.update_heatmap(target_heatmap[i,...], centers, radii, instances['labels'])
            num_dets += len(boxes)

            # box: 3x3 center sampling
            box_samples = []
            indices = []
            for box, (cx, cy) in zip(boxes, centers):
                cxs = [d for d in [cx-1, cx, cx+1] if 0 <= d <= out_w-1]
                cys = [d for d in [cy-1, cy, cy+1] if 0 <= d <= out_h-1]

                new_centers = itertools.product(cxs, cys)
                for cx, cy in new_centers:
                    indices.append(cy*out_w + cx)
                    box_samples.append(box)             # use the same box
                    num_boxes += 1

            box_samples = torch.from_numpy(np.stack(box_samples, axis=0)) * self.stride
            box_samples = box_convert(box_samples, "xywh", "xyxy").to(device)
            pred_boxes = self.gather_and_decode_boxes(box_offsets[i], torch.tensor(indices, device=device))
            box_loss += self.box_loss(pred_boxes, box_samples)

        heatmap_loss = self.heatmap_loss(heatmap, target_heatmap) / max(1, num_dets)
        box_loss = box_loss / max(1, num_boxes)

        return {
            "heatmap": heatmap_loss,
            "box_2d": box_loss,
            "total": heatmap_loss*self.hparams.heatmap_loss_weight + box_loss*self.hparams.box_loss_weight
        }

    # TODO: make this torchscript-able
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/coco.py
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    @staticmethod
    def update_heatmap(heatmap: torch.Tensor, centers: np.ndarray, radii: np.ndarray, labels: Tuple[int]):
        out_h, out_w = heatmap.shape[-2:]
        for (cx, cy), (rx, ry), label in zip(centers, radii, labels):
            # TODO: check CenterNet, mmdet implementation, and CenterNet2
            rx, ry = max(0, round(rx)), max(0, round(ry))
            std_x, std_y = rx/3 + 1/6, ry/3 + 1/6
            
            l, t = min(cx, rx), min(cy, ry)
            r, b = min(out_w - cx, rx+1), min(out_h - cy, ry+1)

            # only gaussian and heatmap are on gpu
            grid_y = torch.arange(-ry, ry+1, device=heatmap.device).view(-1,1)
            grid_x = torch.arange(-rx, rx+1, device=heatmap.device).view(1,-1)

            gaussian = grid_x.square() / (2 * std_x**2) + grid_y.square() / (2 * std_y**2)
            gaussian = torch.exp(-gaussian)
            gaussian[gaussian < torch.finfo(gaussian.dtype).eps * torch.max(gaussian)] = 0

            masked_heatmap = heatmap[label, cy-t:cy+b, cx-l:cx+r]
            masked_gaussian = gaussian[ry-t:ry+b, rx-l:rx+r]
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
        transforms = parse_albumentations_transforms(config['transforms'])
        ds = CocoDetection(config['img_dir'], config['ann_json'], transforms=transforms)

        return DataLoader(
            ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
            shuffle=train, collate_fn=coco_detection_collate_fn, pin_memory=True
        )

    def decode_detections(self, heatmap: torch.Tensor, box_offsets: torch.Tensor, normalize_boxes: bool=False) -> Dict[str, torch.Tensor]:
        """Decode model outputs to detections. Returns: a Dict with keys boxes, scores, labels
        """
        scores, indices, labels = self.get_topk_from_heatmap(heatmap)
        boxes = self.gather_and_decode_boxes(box_offsets, indices, normalize_boxes=normalize_boxes)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }

    def get_topk_from_heatmap(self, heatmap: torch.Tensor):
        """Gather top k detections from heatmap. Batch dim is optional. Returns: scores, indices, labels with dim (N, k)
        """
        batch_size = heatmap.shape[0]
        nms_kernel = self.hparams.nms_kernel

        # 1. pseudo-nms via max pool
        padding = (nms_kernel - 1) // 2
        nms_mask = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=padding) == heatmap
        heatmap = heatmap * nms_mask
        
        # 2. since box regression is shared, we only consider the best candidate at each heatmap location
        heatmap, labels = torch.max(heatmap, dim=1)

        # 3. flatten and get topk
        heatmap = heatmap.view(batch_size, -1)
        labels = labels.view(batch_size, -1)
        scores, indices = torch.topk(heatmap, self.hparams.num_detections)
        labels = torch.gather(labels, dim=-1, index=indices)
    
        return scores, indices, labels

    def gather_and_decode_boxes(self, box_offsets: torch.Tensor, indices: torch.Tensor, normalize_boxes: bool=False) -> torch.Tensor:
        """Gather 2D bounding boxes at given indices.

        Args:
            box_offsets: (N, 4, H, W)
            indices: (N, num_dets)

        Returns:
            boxes: (N, num_dets, 4)
        """
        out_h, out_w = box_offsets.shape[-2:]
        cx = indices % out_w + 0.5
        cy = indices // out_w + 0.5

        # decoded = multiplier x exp(encoded)
        box_offsets = box_offsets.flatten(start_dim=-2)
        if self.hparams.box_log:
            box_offsets = torch.exp(box_offsets)
        box_offsets = box_offsets * self.hparams.box_multiplier     # *= will multiply inplace -> cannot call .backward()
        box_offsets = box_offsets.clamp_min(0)

        # boxes are in output feature maps coordinates
        x1 = cx - torch.gather(box_offsets[...,0,:], dim=-1, index=indices)       # x1 = cx - left
        y1 = cy - torch.gather(box_offsets[...,1,:], dim=-1, index=indices)       # y1 = cy - top
        x2 = cx + torch.gather(box_offsets[...,2,:], dim=-1, index=indices)       # x2 = cx + right
        y2 = cy + torch.gather(box_offsets[...,3,:], dim=-1, index=indices)       # y2 = cy + bottom
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        if normalize_boxes:             # convert to normalized coordinates
            boxes[...,[0,2]] /= out_w
            boxes[...,[1,3]] /= out_h
        else:
            boxes *= self.stride        # convert to input coordinates

        return boxes
