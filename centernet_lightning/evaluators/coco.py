import contextlib
from typing import List, Dict

import torch.distributed as dist
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def gather_and_merge(data: list):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return data

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    merged_data = [x for data in data_list for x in data]
    return merged_data


class CocoEvaluator:
    pred_keys = ("boxes", "scores", "labels")
    target_keys = ("boxes", "labels")
    metric_names = (
        "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "mAR", "AR_small", "AR_medium", "AR_large"
    )
    # sample
    # Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.000
    # Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.000
    # Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.000
    # Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
    # Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
    # Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
    # Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.000
    # Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.000
    # Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.000
    # Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
    # Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
    # Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def update(self, preds: List[Dict[str, np.ndarray]], targets: List[Dict[str, np.ndarray]]):
        """
        Args:
            preds: a list, each is a dict corresponds to 1 image. Keys: boxes (xywh), scores, labels
            targets: a list, each is a dict corresponds to 1 image. Keys: boxes (xywhh), labels
        """
        assert len(preds) == len(targets)
        self.preds.extend(preds)
        self.targets.extend(targets)

    def reset(self):
        self.preds = []
        self.targets = []

    def get_metrics(self):
        preds = gather_and_merge(self.preds)
        targets = gather_and_merge(self.targets)

        image_ids = list(range(len(targets)))          # ensure preds and targets use the same image_ids
        coco_pred = CocoEvaluator.create_coco(preds, image_ids, self.num_classes, prediction=True)
        coco_target = CocoEvaluator.create_coco(targets, image_ids, self.num_classes, prediction=False)

        with contextlib.redirect_stdout(None):
            coco_eval = COCOeval(coco_target, coco_pred, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        metrics = {metric: coco_eval.stats[i] for i, metric in enumerate(self.metric_names)}
        return metrics

    @staticmethod
    def create_coco(detections: List[Dict[str, np.ndarray]], image_ids: List[int], num_classes: int, prediction=False):
        annotations = []
        ann_id = 1

        for img_id, det in zip(image_ids, detections):
            det = {k: v.tolist() for k, v in det.items()}
            for i, (box, label) in enumerate(zip(det["boxes"], det["labels"])):
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0
                }
                if prediction:
                    ann["score"] = det["scores"][i]
                
                annotations.append(ann)
                ann_id += 1

        # mimic COCO.__init__() behavior
        with contextlib.redirect_stdout(None):
            coco = COCO()
            coco.dataset = {
                "images": [{"id": img_id} for img_id in image_ids],
                "annotations": annotations,
                "categories": [{"id": i, "name": i} for i in range(num_classes)]
            }
            coco.createIndex()

        return coco
