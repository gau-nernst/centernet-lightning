from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .coco import COCODataset
from .voc import VOCDataset
from .utils import IMAGENET_MEAN, IMAGENET_STD, collate_detections_with_padding

_dataset_mapper = {
    "coco": COCODataset,
    "voc": VOCDataset
}

_format_mapper = {
    "coco": "coco",
    "voc": "pascal_voc"
}

def build_dataloader(type, dataset_params, dataloader_params, transforms):
    assert type in _dataset_mapper

    transforms = parse_transforms(transforms, format=_format_mapper[type])
    dataset = _dataset_mapper[type](transforms=transforms, **dataset_params)
    dataloader = DataLoader(dataset, collate_fn=collate_detections_with_padding, **dataloader_params)

    return dataloader

def parse_transforms(transforms_cfg, format="coco"):
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
