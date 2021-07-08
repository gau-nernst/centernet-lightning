from typing import Dict

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .coco import COCODataset
from .voc import VOCDataset
from .utils import CollateDetection, IMAGENET_MEAN, IMAGENET_STD

__all__ = ["build_dataset", "build_dataloader"]

_dataset_mapper = {
    "coco": COCODataset,
    "voc": VOCDataset
}

def build_dataset(config):
    dataset_type = config["type"]
    transforms = parse_transforms(config["transforms"]) if "transforms" in config else None

    params = {k:v for k,v in config.items() if k not in ("type", "transforms")}
    dataset = _dataset_mapper[dataset_type](transforms=transforms, **params)
    return dataset

def build_dataloader(config):
    dataset = build_dataset(config["dataset"])
    collate_fn = CollateDetection()
    
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **config["dataloader"])
    return dataloader

def parse_transforms(config: Dict[str, Dict], format="yolo"):
    transforms = []
    for name, params in config.items():
        t = A.__dict__[name](**params)
        transforms.append(t)

    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255))
    transforms.append(ToTensorV2())

    bbox_params = A.BboxParams(format=format, label_fields=["labels"])
    # bbox_params = A.BboxParams(format=format, label_fields=["labels"], min_area=1024, min_visibility=0.1)

    transforms = A.Compose(transforms, bbox_params=bbox_params)
    return transforms
