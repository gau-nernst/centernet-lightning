from typing import Iterable

import numpy as np

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_default_transforms(resize_height=512, resize_width=512, box_format="yolo", label_fields=["labels"]):
    transforms = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        A.Resize(resize_height, resize_width),
        ToTensorV2()
    ]
    bbox_params = A.BboxParams(format=box_format, label_fields=label_fields, min_area=1)

    transforms = A.Compose(transforms, bbox_params=bbox_params)
    return transforms

def get_default_detection_transforms(box_format="yolo"):
    """ImageNet normalization and resize to 512 x 512
    """
    transforms = get_default_transforms(resize_height=512, resize_width=512, box_format=box_format, label_fields=["labels"])
    return transforms

def get_default_tracking_transforms(box_format="yolo"):
    """ImageNet normalization and resize to 1088 x 608 (close to 16:9 and divisible by 32)
    """
    transforms = get_default_transforms(resize_height=608, resize_width=1088, box_format=box_format, label_fields=["labels", "ids"])
    return transforms

class CollateDetection:

    def __call__(self, batch: Iterable):
        """Receive a batch of items, each contains the following keys:
            - image: torch.Tensor image in CHW format
            - bboxes: nested list of bounding boxes
            - labels: a list of labels
        """
        batch_size = len(batch)
        max_length = max(len(x["labels"]) for x in batch)

        bboxes = np.zeros(shape=(batch_size, max_length, 4), dtype=np.float32)
        labels = np.zeros(shape=(batch_size, max_length), dtype=np.int32)
        mask = np.zeros(shape=(batch_size, max_length), dtype=np.uint8)
        
        for b, item in enumerate(batch):
            num_detections = len(item["labels"])
            if num_detections > 0:
                bboxes[b,:num_detections] = item["bboxes"]
                labels[b,:num_detections] = item["labels"]
                mask[b,:num_detections] = 1

        image = torch.stack([x["image"] for x in batch], dim=0)   # NCHW
        bboxes = torch.from_numpy(bboxes)
        labels = torch.from_numpy(labels)
        mask = torch.from_numpy(mask)

        output = {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "mask": mask
        }
        return output

class CollateTracking:

    def __call__(self, batch: Iterable):
        """Receive a batch of items, each contains the following keys:
            - image: torch.Tensor image in CHW format
            - bboxes: nested list of bounding boxes
            - labels: a list of labels
            - ids: a list of track ids
        """
        batch_size = len(batch)
        max_length = max(len(x["labels"]) for x in batch)

        bboxes = np.zeros(shape=(batch_size, max_length, 4), dtype=np.float32)
        labels = np.zeros(shape=(batch_size, max_length), dtype=np.int32)
        ids = np.zeros(shape=(batch_size, max_length), dtype=np.int32)
        mask = np.zeros(shape=(batch_size, max_length), dtype=np.uint8)
        
        for b, item in enumerate(batch):
            num_detections = len(item["labels"])
            if num_detections > 0:
                bboxes[b,:num_detections] = item["bboxes"]
                labels[b,:num_detections] = item["labels"]
                ids[b,:num_detections] = item["ids"]
                mask[b,:num_detections] = 1

        image = torch.stack([x["image"] for x in batch], dim=0)   # NCHW
        bboxes = torch.from_numpy(bboxes)
        labels = torch.from_numpy(labels)
        ids = torch.from_numpy(ids)
        mask = torch.from_numpy(mask)

        output = {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "ids": ids,
            "mask": mask
        }
        return output
