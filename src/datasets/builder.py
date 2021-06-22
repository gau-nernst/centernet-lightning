from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .coco import COCODataset
from .voc import VOCDataset
from .utils import CollateDetectionsCenterNet, IMAGENET_MEAN, IMAGENET_STD

__all__ = ["build_dataset", "build_dataloader"]

_dataset_mapper = {
    "coco": COCODataset,
    "voc": VOCDataset
}

def build_dataset(type, dataset_params, transforms = None, **kwargs):
    assert type in _dataset_mapper

    if transforms is not None:
        transforms = parse_transforms(transforms)
    dataset = _dataset_mapper[type](transforms=transforms, **dataset_params)
    return dataset

def build_dataloader(model, type, dataset_params, dataloader_params, transforms = None, **kwargs):
    """A CenterNet model is required to build the dataloader, since it needs to know the params to create target heatmap
    """
    dataset = build_dataset(type, dataset_params, transforms=transforms)
    
    img_shape = dataset[0]["image"].shape
    heatmap_shape = (model.num_classes, img_shape[1]//model.output_stride, img_shape[2]//model.output_stride)

    collate_fn = CollateDetectionsCenterNet(heatmap_shape, heatmap_method=model.hparams["heatmap_method"])
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_params)
    return dataloader

def parse_transforms(transforms_cfg, format="yolo"):
    transforms = []
    for x in transforms_cfg:
        transf = A.__dict__[x["name"]](**x["params"])
        transforms.append(transf)

    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255))
    transforms.append(ToTensorV2())
    transforms = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format=format, min_area=1024, min_visibility=0.1, label_fields=["labels"])
    )
    return transforms
