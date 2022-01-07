from typing import List
from functools import partial

import torch
from torch import nn
from torchmetrics.detection.map import MAP

from .meta import MetaCenterNet
from .heads import HeatmapHead, Box2DHead
from ..utils import convert_cxcywh_to_xywh, convert_x1y1x2y2_to_xywh


class CenterNet(MetaCenterNet):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module, 
        neck: nn.Module,
        
        heatmap_prior: float=0.1,
        heatmap_method: str="cornernet",
        heatmap_loss: str="cornetnet_focal",
        
        box_loss: str="l1",
        log_box: bool=False,
        box_multiplier: float=1,

        **kwargs
        ):
        heads = {
            "heatmap": partial(
                HeatmapHead, 
                num_classes=num_classes, heatmap_prior=heatmap_prior,
                target_method=heatmap_method, loss_function=heatmap_loss
            ),
            "box_2d": partial(Box2DHead, loss_function=box_loss, log_box=log_box, box_multiplier=box_multiplier)
        }
        super().__init__(backbone, neck, heads, **kwargs)
        self.metric = MAP()

    def validation_step(self, batch, batch_idx):
        # batch keys: image, boxes, labels
        outputs = self.get_encoded_outputs(batch["image"])
        losses = self.compute_loss(outputs, batch)
        for k, v in losses.items():
            self.log(f"val/{k}_loss", v)
        
        # target box format?
        # convert dict to list
        preds = self.gather_detections(outputs["heatmap"].sigmoid(), outputs["box_2d"])
        preds = [{k: v[i]} for i, (k, v) in enumerate(preds.items())]    
        target = [{k: batch[k][i] for k in ("boxes", "labels")} for i in range(len(batch["boxes"]))]
        
        self.metric.update(preds, target)
    
    def gather_detections(
        self,
        heatmap: torch.Tensor,
        box_offsets: torch.Tensor,
        num_detections: int=300,
        nms_kernel: int=3,
        normalize_boxes: bool=False,
        img_widths: List[int]=None,
        img_heights: List[int]=None
        ):
        """Decode model outputs for detection task

        Args
            heatmap: heatmap output
            box_2d: box_2d output
            num_detections: number of detections to return. Default is 100
            nms_kernel: the kernel used for max pooling (pseudo-nms). Larger values will reduce false positives. Default is 3 (original paper)
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
        """
        if not normalize_boxes:
            assert img_widths is not None and img_heights is not None
        
        scores, indices, labels = self.heads["heatmap"].gather_topk(heatmap, nms_kernel=nms_kernel, num_detections=num_detections)
        boxes = self.heads["box_2d"].gather_and_decode(box_offsets, indices, normalize_boxes=True)
        if not normalize_boxes:
            boxes[...,[0,2]] *= img_widths
            boxes[...,[1,3]] *= img_heights

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
