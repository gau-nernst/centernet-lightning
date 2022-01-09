from typing import Any, List, Dict, Tuple, Union
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.ops import box_convert
from torchmetrics.detection.map import MAP
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .meta import BaseHead, MetaCenterNet
from ..losses import heatmap_losses, box_losses
from ..datasets.coco import CocoDetection, collate_fn

from vision_toolbox import backbones, necks

class HeatmapRenderer:
    def __init__(self, stride=4, **kwargs):
        self.stride = stride
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_radius(self, width, height):
        return 1, 1

    # TODO: make this torchscript-able
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/coco.py
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    def __call__(self, heatmap: torch.Tensor, boxes: Tuple[Tuple[int]], labels: Tuple[int]) -> torch.Tensor:
        """Render target heatmap for a batch of images
        """
        out_h, out_w = heatmap.shape[-2:]
        for box, label in zip(boxes, labels):
            # scale up to heatmap dimensions
            x, y, w, h = [i / self.stride for i in box]
            cx = round(x + w/2)
            cy = round(y + h/2)
            
            r_x, r_y = self.get_radius(w, h)
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

        return heatmap


class FixedRadiusRenderer(HeatmapRenderer):
    def get_radius(self, width, height):
        return self.radius, self.radius


class CornerNetRenderer(HeatmapRenderer):
    """Get radius for the Gaussian kernel. From CornerNet https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    """
    # https://github.com/princeton-vl/CornerNet/issues/110
    def get_radius(self, width, height):
        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - self.min_overlap) / (1 + self.min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - self.min_overlap) * width * height
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * self.min_overlap
        b3 = -2 * self.min_overlap * (height + width)
        c3 = (self.min_overlap - 1) * width * height
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        r = min(r1, r2, r3)
        return r, r


class TTFNetRenderer(HeatmapRenderer):
    def get_radius(self, width, height):
        return width/2 * self.alpha, height/2 * self.alpha


_heatmap_renderers = {
    "fixedradius": FixedRadiusRenderer,
    "cornernet": CornerNetRenderer,
    "ttfnet": TTFNetRenderer
}

class HeatmapHead(BaseHead):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stride: int=4,
        heatmap_prior: float=0.1,

        heatmap_method: str="cornernet",
        radius: float=1,                # for fixed radius
        min_overlap: float=0.3,         # for cornetnet
        alpha: float=0.54,              # for ttfnet
        loss_function: str="CornerNetFocalLoss",
        nms_kernel: int=3,
        num_detections: int=300,

        **base_head_kwargs
        ):
        init_bias = math.log(heatmap_prior/(1-heatmap_prior))
        super().__init__(in_channels, num_classes, init_bias=init_bias, **base_head_kwargs)
        
        self.renderer: HeatmapRenderer = _heatmap_renderers[heatmap_method](stride=stride, radius=radius, min_overlap=min_overlap, alpha=alpha)
        self.loss_function = heatmap_losses.__dict__[loss_function]()
        self.nms_kernel = nms_kernel
        self.num_detections = num_detections

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Union[List, int]]]) -> torch.Tensor:
        heatmap = outputs["heatmap"]

        target_heatmap = torch.zeros_like(heatmap)
        num_dets = 0
        for i, instances in enumerate(targets):
            self.renderer(target_heatmap[i,...], instances["boxes"], instances["labels"])
            num_dets += len(instances["labels"])

        loss = self.loss_function(heatmap, target_heatmap) / max(1, num_dets)
        return loss

    def gather_topk(self, heatmap: torch.Tensor):
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

        if squeeze:
            scores = scores.squeeze(0)
            indices = indices.squeeze(0)
            labels = labels.squeeze(0)

        return scores, indices, labels


class Box2DHead(BaseHead):
    def __init__(self, in_channels: int, stride: int=4, loss_function: str="L1Loss", log_box: bool=False, box_multiplier: float=1., **base_head_kwargs):
        out_channels = 4
        super().__init__(in_channels, out_channels, **base_head_kwargs)
        self.stride = stride
        self.loss_function = box_losses.__dict__[loss_function](reduction="sum")
        self.log_box = log_box
        self.box_multiplier = box_multiplier

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Tuple[Dict[str, Union[Tuple, int]]]) -> torch.Tensor:
        box_offsets = outputs["box_2d"]           # (N, 4, H, W)
        out_w = box_offsets.shape[-1]

        loss = torch.tensor(0., dtype=box_offsets.dtype, device=box_offsets.device)
        num_dets = 0

        for i, instances in enumerate(targets):
            if len(instances["boxes"]):     # skip image without boxes
                # 1. convert target boxes to xyxy and get center points
                img_boxes = box_convert(torch.tensor(instances["boxes"]), "xywh", "xyxy")
                cx = (img_boxes[...,0] + img_boxes[...,2]) / 2 / self.stride        # center points in output feature map coordinates
                cy = (img_boxes[...,1] + img_boxes[...,3]) / 2 / self.stride

                # 2. gather training samples. only center point
                # TODO: 3x3 square
                indices = cy.long() * out_w + cx.long()
                pred_boxes = self.gather_and_decode(box_offsets[i], indices.to(box_offsets.device))

                # 3. apply loss
                loss += self.loss_function(pred_boxes, img_boxes.to(box_offsets.device))
                num_dets += len(instances["boxes"])

        loss = loss.sum() / max(1, num_dets)
        return loss

    def gather_and_decode(self, box_offsets: torch.Tensor, indices: torch.Tensor, normalize_boxes: bool=False) -> torch.Tensor:
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
        if self.log_box:
            box_offsets = torch.exp(box_offsets)
        if self.box_multiplier > 1:
            box_offsets *= self.box_multiplier
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
            boxes *= self.stride        # convert to input coordinates

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
        **kwargs
        ):
        backbone: backbones.BaseBackbone = backbones.__dict__[backbone](pretrained=pretrained_backbone)
        
        if neck_config is None: neck_config = {}
        neck = necks.__dict__[neck](backbone.get_out_channels(), **neck_config)
        stride = backbone.stride // neck.stride

        if heatmap_config is None: heatmap_config = {}
        if box2d_config is None: box2d_config = {}
        heads = {
            "heatmap": HeatmapHead(neck.out_channels, num_classes, stride=stride, **heatmap_config),
            "box_2d": Box2DHead(neck.out_channels, stride=stride, **box2d_config)
        }
        super().__init__(backbone, neck, heads, stride=stride, **kwargs)
        self.metric = MAP()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.get_encoded_outputs(images)

        preds = self.gather_detections(outputs["heatmap"].sigmoid(), outputs["box_2d"])
        preds = [{k: v[i] for k, v in preds.items()} for i in range(len(targets))]  # convert dict to list
        
        # extract required keys
        targets = [{
            "boxes": box_convert(torch.tensor(x["boxes"], device=self.device), "xywh", "xyxy"),
            "labels": torch.tensor(x["labels"], device=self.device)
        } for x in targets]

        # TODO: scale to original image size to get correct AP small, medium, large?
        self.metric.update(preds, targets)
    
    def get_dataloader(self, train=True):
        if train:
            config = self.hparams.train_data
            min_area = 1
            shuffle = True
        else:
            config = self.hparams.val_data
            min_area = -1        # don't remove boxes for validation set
            shuffle = False

        ts = []
        for t in config["transforms"]:
            t_fn = A.__dict__[t["name"]]
            ts.append(t_fn(**t["init_args"]) if "init_args" in t else t_fn())
        ts.append(ToTensorV2())
        
        transforms = A.Compose(ts, bbox_params=dict(format="coco", label_fields=["labels"], min_area=min_area))
        ds = CocoDetection(config["img_dir"], config["ann_json"], transforms=transforms)

        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle, collate_fn=collate_fn, pin_memory=True)

    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def gather_detections(self, heatmap: torch.Tensor, box_offsets: torch.Tensor, normalize_boxes: bool=False) -> Dict[str, torch.Tensor]:
        """Decode model outputs for detection task

        Args:
            heatmap: heatmap output
            box_offsets: box_2d output
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
            img_widths
            img_heights
        
        Returns: a Dict with keys boxes, scores, labels
        """
        scores, indices, labels = self.heads["heatmap"].gather_topk(heatmap)
        boxes = self.heads["box_2d"].gather_and_decode(box_offsets, indices, normalize_boxes=normalize_boxes)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }

    # @torch.inference_mode()
    # def inference_detection2d(self, data_dir, img_names, batch_size=4, num_detections=100, nms_kernel=3, save_path=None, score_threshold=0):
    #     """Run detection on a folder of images
    #     """
    #     transforms = A.Compose([
    #         A.Resize(height=512, width=512),
    #         A.Normalize(),
    #         ToTensorV2()
    #     ])
    #     dataset = InferenceDataset(data_dir, img_names, transforms=transforms, file_ext=".jpg")
    #     dataloader = DataLoader(dataset, batch_size=batch_size)

    #     all_detections = {
    #         "bboxes": [],
    #         "labels": [],
    #         "scores": []
    #     }

    #     self.eval()
    #     for batch in tqdm(dataloader):
    #         img_widths = batch["original_width"].clone().numpy().reshape(-1,1,1)
    #         img_heights = batch["original_height"].clone().numpy().reshape(-1,1,1)

    #         heatmap, box_2d = self(batch["image"].to(self.device))
    #         detections = self.gather_detections(heatmap, box_2d, num_detections=num_detections, nms_kernel=nms_kernel, normalize_boxes=True)
    #         detections = {k: v.cpu().float().numpy() for k,v in detections.items()}

    #         detections["bboxes"][...,[0,2]] *= img_widths
    #         detections["bboxes"][...,[1,3]] *= img_heights

    #         for k, v in detections.items():
    #             all_detections[k].append(v)

    #     all_detections = {k: np.concatenate(v, axis=0) for k,v in all_detections.items()}
        
    #     if save_path is not None:
    #         bboxes = detections["bboxes"].tolist()
    #         labels = detections["labels"].tolist()
    #         scores = detections["scores"].tolist()

    #         detections_to_coco_results(range(len(img_names)), bboxes, labels, scores, save_path, score_threshold=score_threshold)

    #     return all_detections
