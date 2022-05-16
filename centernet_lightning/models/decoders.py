from typing import Any, Dict, Tuple, Optional
import math
import itertools

# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# from torch.utils.data import DataLoader, ConcatDataset
# from torchvision.ops import box_convert, batched_nms
# import albumentations as A

# from .meta import Head, GenericLightning
# from ..losses import heatmap_losses, box_losses
# from ..datasets.coco import CocoDetection, coco_detection_collate_fn
# from ..eval.coco import CocoEvaluator


# class CenterNet(GenericLightning):
#     def __init__(
#         self,
#         num_classes: int,
#         backbone: str,
#         pretrained_backbone: bool=False,
#         neck: str='FPN',
#         neck_config: Dict[str, Any]=None,
#         head_config: Dict[str, Any]=None,

#         # box params
#         box_init_bias: float=None,
#         box_loss: str='L1Loss',
#         box_loss_weight: float=0.1,
#         box_log: bool=False,
#         box_multiplier: float=1.,

#         # heatmap params
#         heatmap_prior: float=0.01,
#         heatmap_loss: str='CornerNetFocalLoss',
#         heatmap_loss_weight: float=1.,
#         heatmap_target: str='cornernet',
#         heatmap_target_params: Dict[str, float]=None,

#         # inference config
#         nms_kernel: int=3,
#         num_detections: int=100,

#         # data
#         train_data: Dict[str, Any]=None,
#         val_data: Dict[str, Any]=None,

#         optimizer_config: Dict[str, Any]=None
#     ):
#         heads = {
#             'heatmap': {'out_channels': num_classes, 'init_bias': math.log(heatmap_prior/(1-heatmap_prior))},
#             'box_2d': {'out_channels': 4, 'init_bias': box_init_bias}
#         }
#         if optimizer_config is None:
#             optimizer_config = {}
#         super().__init__(
#             backbone, pretrained_backbone, neck, heads,
#             neck_config=neck_config, head_config=head_config, **optimizer_config
#         )
#         self.save_hyperparameters(ignore='optimizer_config')

#         self.num_classes = num_classes
#         self.evaluator = CocoEvaluator(num_classes)

#         self.heatmap_loss = heatmap_losses.__dict__[heatmap_loss](reduction='sum')
#         self.box_loss = box_losses.__dict__[box_loss](reduction='sum')
#         if heatmap_target_params is None:
#             heatmap_target_params = {}

#     def compute_loss(self, outputs, targets, stride=None):
#         if stride is None:
#             stride = self.stride

#         heatmap = outputs['heatmap']
#         box_offsets = outputs['box_2d']
#         out_h, out_w = heatmap.shape[-2:]
#         dtype = heatmap.dtype
#         device = self.device

#         num_dets = num_boxes = 0
#         target_heatmap = torch.zeros_like(heatmap)
#         box_loss = torch.tensor(0., dtype=dtype, device=device)
#         for i, instances in enumerate(targets):
#             if len(instances['labels']) == 0:
#                 continue


#             CenterNet.update_heatmap(target_heatmap[i], centers, radii, instances['labels'])
#             num_dets += len(boxes)

#             # box: 3x3 center sampling
#             box_samples, indices = [], []
#             for (x, y, w, h), (cx, cy) in zip(instances['boxes'], centers):
#                 box = [x, y, x+w, y+h]
#                 cxs = [d for d in [cx-1, cx, cx+1] if 0 <= d <= out_w-1]
#                 cys = [d for d in [cy-1, cy, cy+1] if 0 <= d <= out_h-1]

#                 new_centers = itertools.product(cxs, cys)
#                 for cx, cy in new_centers:
#                     indices.append(cy*out_w + cx)
#                     box_samples.append(box)             # this is the original box, in input image scale
#                     num_boxes += 1

#             pred_boxes = CenterNet.gather_and_decode_boxes(
#                 box_offsets[i], torch.tensor(indices, device=device),
#                 box_log=self.hparams.box_log, box_multiplier=self.hparams.box_multiplier, stride=stride
#             )
#             box_loss += self.box_loss(pred_boxes, torch.tensor(box_samples, device=device))

#         heatmap_loss = self.heatmap_loss(heatmap, target_heatmap) / max(1, num_dets)
#         box_loss = box_loss / max(1, num_boxes)

#         return {
#             'heatmap': heatmap_loss,
#             'box_2d': box_loss,
#             'total': heatmap_loss*self.hparams.heatmap_loss_weight + box_loss*self.hparams.box_loss_weight
#         }

#     def validation_step(self, batch, batch_idx):
#         images, targets = batch
#         if self.hparams.channels_last:
#             images = images.to(memory_format=torch.channels_last)
#         outputs = self.model(images)
#         preds = CenterNet.decode_detections(outputs['heatmap'].sigmoid(), outputs['box_2d'])

#         preds['boxes'] = box_convert(preds['boxes'], 'xyxy', 'xywh')                    # coco box format
#         preds = {k: v.cpu().numpy() for k, v in preds.items()}                          # convert to numpy
#         preds = [{k: v[i] for k, v in preds.items()} for i in range(images.shape[0])]   # convert to list of images

#         # outputs = self.model.multilevel_forward(images)
#         # preds = {k: [] for k in ('boxes', 'scores', 'labels')}
#         # for out in outputs:
#         #     for k, v in CenterNet.decode_detections(out['heatmap'].sigmoid(), out['box_2d']):
#         #         preds[k].append(v)

#         # preds = {k: torch.concat(v, dim=1) for k, v in preds.items()}

#         targets = [{k: np.array(target[k]) for k in ('boxes', 'labels')} for target in targets]     # filter keys and convert to numpy array
#         self.evaluator.update(preds, targets)

#     def validation_epoch_end(self, outputs):
#         metrics = self.evaluator.get_metrics()
#         self.evaluator.reset()
#         for k, v in metrics.items():
#             self.log(f'val/{k}', v)

#     def get_dataloader(self, train=True):
#         config = self.hparams.train_data if train else self.hparams.val_data
#         transforms = A.load(config['transforms'], data_format='yaml')
#         ds = CocoDetection(config['img_dir'], config['ann_json'], transforms=transforms)
#         return DataLoader(
#             ds, batch_size=config['batch_size'], num_workers=config['num_workers'],
#             shuffle=train, collate_fn=coco_detection_collate_fn, pin_memory=True
#         )


class CenterNetDecoder:
    def __init__(self, box_log: bool = False, box_multiplier: float = 16.0):
        self.box_log = box_log
        self.box_multiplier = box_multiplier

    @staticmethod
    def get_topk_from_heatmap(
        heatmap: torch.Tensor,
        pseudo_nms: bool = True,
        nms_kernel: int = 3,
        k: int = 100,
    ):
        """Gather top k detections from heatmap. Batch dim is optional.

        Returns: scores, indices, labels with dim (N, k)
        """
        batch_size = heatmap.shape[0]

        # 1. pseudo-nms via max pool
        if pseudo_nms:
            padding = (nms_kernel - 1) // 2
            pooled_heatmap = F.max_pool2d(
                heatmap, kernel_size=nms_kernel, stride=1, padding=padding
            )
            nms_mask = pooled_heatmap == heatmap
            heatmap = heatmap * nms_mask
        # get best candidate at each heatmap location, since box regression is shared
        heatmap, labels = torch.max(heatmap, dim=1)

        # 3. flatten and get topk
        heatmap = heatmap.view(batch_size, -1)
        labels = labels.view(batch_size, -1)
        scores, indices = torch.topk(heatmap, k)
        labels = torch.gather(labels, dim=-1, index=indices)
        return scores, indices, labels

    def gather_and_decode_boxes(
        self,
        box_offsets: torch.Tensor,
        indices: torch.Tensor,
        normalize_boxes: bool = False,
        stride: int = 4,
    ) -> torch.Tensor:
        """Gather 2D bounding boxes at given indices.

        Args:
            box_offsets: (N, 4, H, W)
            indices: (N, num_dets)

        Returns:
            boxes: (N, num_dets, 4)
        """
        out_h, out_w = box_offsets.shape[-2:]
        cx = torch.remainder(indices, out_w) + 0.5
        cy = torch.div(indices, out_w, rounding_mode="floor") + 0.5

        # decoded = multiplier x exp(encoded)
        box_offsets = box_offsets.flatten(start_dim=-2)
        if self.box_log:
            box_offsets = torch.exp(box_offsets)
        box_offsets = box_offsets * self.box_multiplier
        box_offsets = F.relu(box_offsets, inplace=True)

        # boxes are in output feature maps coordinates
        # boxes = torch.stack((cx,cy,cx,cy), dim=-1)
        # boxes[...,:2] -= torch.gather(box_offsets[...,:2,:], dim=-1, index=indices)
        # boxes[...,2:] += torch.gather(box_offsets[...,2:,:], dim=-1, index=indices)

        x1 = cx - torch.gather(
            box_offsets[..., 0, :], dim=-1, index=indices
        )  # x1 = cx - left
        y1 = cy - torch.gather(
            box_offsets[..., 1, :], dim=-1, index=indices
        )  # y1 = cy - top
        x2 = cx + torch.gather(
            box_offsets[..., 2, :], dim=-1, index=indices
        )  # x2 = cx + right
        y2 = cy + torch.gather(
            box_offsets[..., 3, :], dim=-1, index=indices
        )  # y2 = cy + bottom
        boxes = torch.stack((x1, y1, x2, y2), dim=-1)

        if normalize_boxes:  # convert to normalized coordinates
            boxes[..., [0, 2]] /= out_w
            boxes[..., [1, 3]] /= out_h
        else:
            boxes *= stride  # convert to input coordinates
        return boxes

    def decode_detections(
        self,
        heatmap: torch.Tensor,
        box_offsets: torch.Tensor,
        topk_kwargs: Dict[str, Any] = None,
        box_kwargs: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode model outputs to detections. Returns: a Dict with keys boxes, scores, labels"""
        if topk_kwargs is None:
            topk_kwargs = {}
        if box_kwargs is None:
            box_kwargs = {}

        scores, indices, labels = self.get_topk_from_heatmap(heatmap, **topk_kwargs)
        boxes = self.gather_and_decode_boxes(box_offsets, indices, **box_kwargs)
        return {"boxes": boxes, "scores": scores, "labels": labels}
