import contextlib
from collections import namedtuple
from typing import List, Optional, TypedDict, TypeVar, Union

import numpy as np
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

T = TypeVar("T")


class _PredictedDetection(TypedDict):
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


class _GroundTruthDetection(TypedDict):
    boxes: np.ndarray
    labels: np.ndarray


CocoMetrics = namedtuple(
    "CocoMetrics",
    [
        "mAP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "mAR",
        "AR_small",
        "AR_medium",
        "AR_large",
    ],
)


def gather_and_merge_list(data: List[T]) -> List[T]:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return data

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    merged_data = [x for data in data_list for x in data]
    return merged_data


class CocoEvaluator:
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
    def __init__(self):
        self.reset()

    def update(
        self,
        predictions: List[_PredictedDetection],
        targets: List[_GroundTruthDetection],
    ):
        """
        Args:
            predictions: a list, each is a dict corresponds to 1 image. Keys: boxes (xywh), scores, labels
            targets: a list, each is a dict corresponds to 1 image. Keys: boxes (xywh), labels
        """
        assert len(predictions) == len(targets)
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def reset(self) -> None:
        self.predictions: List[_PredictedDetection] = []
        self.targets: List[_GroundTruthDetection] = []
        self._results = None

    def compute(self) -> CocoMetrics:
        if self._results is not None:
            return self._results

        predictions = gather_and_merge_list(self.predictions)
        targets = gather_and_merge_list(self.targets)

        image_ids = list(
            range(len(targets))
        )  # ensure preds and targets use the same image_ids
        coco_predictions = _create_coco(predictions, image_ids)
        coco_targets = _create_coco(targets, image_ids)

        with contextlib.redirect_stdout(None):
            coco_eval = COCOeval(coco_targets, coco_predictions, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        results = CocoMetrics(*coco_eval.stats)
        self._results = results
        return results


def _create_coco(
    detections: List[Union[_PredictedDetection, _GroundTruthDetection]],
    image_ids: Optional[List[int]] = None,
) -> COCO:
    if image_ids is None:
        image_ids = list(range(len(detections)))
    annotations = []
    ann_id = 1
    num_classes = 0

    for img_id, det in zip(image_ids, detections):
        boxes, labels, scores = det["boxes"], det["labels"], det.get("scores", None)
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert labels.ndim == 1
        assert labels.shape[0] == boxes.shape[0]
        if scores is not None:
            assert scores.ndim == 1
            assert scores.shape[0] == labels.shape[0]

        for i, (box, label) in enumerate(zip(boxes, labels)):
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": label,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
            if scores is not None:
                ann["score"] = scores[i]

            annotations.append(ann)
            ann_id += 1
            num_classes = max(num_classes, label + 1)

    # mimic COCO.__init__() behavior
    with contextlib.redirect_stdout(None):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": img_id} for img_id in image_ids],
            "annotations": annotations,
            "categories": [{"id": i, "name": i} for i in range(num_classes)],
        }
        coco.createIndex()

    return coco
