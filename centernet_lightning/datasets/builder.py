from typing import Dict

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .coco import COCODataset
from .voc import VOCDataset
from .crowdhuman import CrowdHumanDataset
from .mot import MOTTrackingDataset
from .kitti import KITTITrackingDataset
from .detection_for_tracking import DetectionForTracking
from .utils import CollateDetection, CollateTracking, IMAGENET_MEAN, IMAGENET_STD

__all__ = ["build_dataset", "build_dataloader"]

_dataset_mapper = {
    "coco": COCODataset,
    "voc": VOCDataset,
    "crowdhuman": CrowdHumanDataset,
    "mot-tracking": MOTTrackingDataset,
    "kitti-tracking": KITTITrackingDataset
}

def build_dataset(config):
    dataset_type = config["type"]
    task = "tracking" if dataset_type.endswith("-tracking") else "detection"
    transforms = parse_transforms(config["transforms"], task=task) if "transforms" in config else None

    params = {k:v for k,v in config.items() if k not in ("type", "transforms", "detection_for_tracking")}
    dataset = _dataset_mapper[dataset_type](transforms=transforms, **params)
    
    # use detection dataset for tracking task
    if "detection_for_tracking" in config and config["detection_for_tracking"]:
        dataset = DetectionForTracking(dataset)

    return dataset

def build_dataloader(config):
    dataset = build_dataset(config["dataset"])
    collate_fn = CollateDetection() if isinstance(dataset, (COCODataset, VOCDataset, CrowdHumanDataset)) else CollateTracking()
    
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **config["dataloader"])
    return dataloader

def parse_transforms(config: Dict[str, Dict], format="yolo", task="detection"):
    transforms = []
    for t_config in config:
        t = A.__dict__[t_config["name"]](**t_config["params"])
        transforms.append(t)

    # transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255))
    transforms.append(ToTensorV2())

    label_fields = ["labels", "ids"] if task == "tracking" else ["labels"]
    bbox_params = A.BboxParams(format=format, label_fields=label_fields, min_area=1)

    transforms = A.Compose(transforms, bbox_params=bbox_params)
    return transforms
