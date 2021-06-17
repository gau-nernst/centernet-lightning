import warnings
from typing import Dict, Iterable
import os
import pickle
import json
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def prepare_coco_detection(data_dir: str, coco_version: str):
    ann_dir = os.path.join(data_dir, "annotations")
    ann_file = os.path.join(ann_dir, f"instances_{coco_version}.json")

    save_dir = os.path.join(ann_dir, coco_version)
    os.makedirs(save_dir, exist_ok=True)
    detection_file = os.path.join(save_dir, "detections.pkl")
    label_to_name_file = os.path.join(save_dir, "label_to_name.json")
    label_to_id_file = os.path.join(save_dir, "label_to_id.json")

    # if already exist on disk, don't do anything
    if os.path.exists(detection_file) and os.path.exists(label_to_name_file) and os.path.exists(label_to_id_file):
        return

    # extract only bboxes and ids data, otherwise train set annotations is too large
    coco = COCO(ann_file)
    categories = OrderedDict(coco.cats)
    
    label_to_id = {0: 0}    # 0 is background class
    label_to_name = {0: "background"}
    for i, v in enumerate(categories.values()):
        label_to_id[i+1] = v["id"]
        label_to_name[i+1] = v["name"]
    id_to_label = {v: k for k, v in label_to_id.items()}

    # save to disk
    with open(label_to_name_file, "w") as f:
        json.dump(label_to_name, f)
    with open(label_to_id_file, "w") as f:
        json.dump(label_to_id, f)

    img_ids = coco.getImgIds()                          # list of all image ids
    img_info = coco.loadImgs(img_ids)                   # list of img info, each is a dict
    img_names = [x["file_name"] for x in img_info]      # we only need file_name to open the images

    annotate_ids = [coco.getAnnIds(imgIds=x) for x in img_ids]      # get annotations for each image
    annotates = [coco.loadAnns(ids=x) for x in annotate_ids]        

    bboxes = []
    labels = []
    for ann in annotates:       # outer loop is loop over images
        img_bboxes = []
        img_labels = []
        for detection in ann:   # inner loop is loop over detections in an image
            bbox = detection["bbox"]
            cat_id = detection["category_id"]

            bbox[2] = max(bbox[2], 1)   # clip width and height to 1
            bbox[3] = max(bbox[3], 1)
        
            img_bboxes.append(bbox)
            img_labels.append(id_to_label[cat_id])

        bboxes.append(img_bboxes)
        labels.append(img_labels)

    detection = {
        "img_ids": img_ids,
        "img_names": img_names,
        "bboxes": bboxes,
        "labels": labels
    }
    with open(detection_file, "wb") as f:
        pickle.dump(detection, f)          # save to disk

    del coco

class COCODataset(Dataset):
    def __init__(self, data_dir: str, coco_version: str, transforms: A.Compose = None):
        super(COCODataset, self).__init__()
        detection_file = os.path.join(data_dir, "annotations", coco_version, "detections.pkl")
        
        with open(detection_file, "rb") as f:
            detection = pickle.load(f)

        if transforms == None:
            warnings.warn("transforms is not specified. Default to normalize with ImageNet and resize to 512x512")
            # use Albumentation resize to handle bbox resizing also
            # centernet resize input to 512 and 512
            transforms = A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255),
                A.Resize(512, 512),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=["labels"]))

        self.img_dir = os.path.join(data_dir, coco_version)
        self.img_names = detection["img_names"]
        self.bboxes = detection["bboxes"]
        self.labels = detection["labels"]
        self.transforms = transforms

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = self.bboxes[index]
        labels = self.labels[index]

        # self.transforms is an Albumentations Transform instance
        # Albumentations will handle transforming the bounding boxes also
        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
        img = augmented["image"]
        bboxes = augmented["bboxes"]
        labels = augmented["labels"]

        for i, box in enumerate(bboxes):
            x, y, w, h = box
            cx = x + w / 2
            cy = y + h / 2
            bboxes[i] = [cx, cy, w, h]

        item = {
            "image": img,
            "bboxes": bboxes,
            "labels": labels
        }
        return item

    def __len__(self):
        return len(self.img_names)

class COCODataModule(pl.LightningDataModule):
    """Lightning Data Module, used for training
    """
    def __init__(self, train: Dict, validation: Dict = None, test: Dict = None):
        super().__init__()
        self.train_cfg = train
        self.val_cfg = validation
        self.test_cfg = test

        self.train_transforms = self._parse_transforms(train["transforms"])
        self.val_transforms = self._parse_transforms(validation["transforms"]) if validation else None
        self.test_transforms = self._parse_transforms(test["transforms"]) if test else None

    def _parse_transforms(self, transforms_cfg):
        transforms = []
        for x in transforms_cfg:
            transf = A.__dict__[x["name"]](**x["params"])
            transforms.append(transf)

        transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255))
        transforms.append(ToTensorV2())
        transforms = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=["labels"])
        )
        return transforms

    def prepare_data(self):
        prepare_coco_detection(self.train_cfg["data_dir"], self.train_cfg["coco_version"])
        prepare_coco_detection(self.val_cfg["data_dir"], self.val_cfg["coco_version"]) if self.val_cfg else None

    def setup(self, stage: str):
        if stage in (None, "fit"):
            self.coco_train = COCODataset(
                self.train_cfg["data_dir"],
                self.train_cfg["coco_version"],
                transforms=self.train_transforms
            )
            self.coco_val = COCODataset(
                self.val_cfg["data_dir"],
                self.val_cfg["coco_version"],
                transforms=self.val_transforms
            )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.coco_train,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_detections_with_padding,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.coco_val,
            batch_size=self.val_cfg["batch_size"],
            shuffle=False,
            num_workers=4,
            collate_fn=collate_detections_with_padding,
            pin_memory=True
        )
        return val_dataloader

def collate_detections_with_padding(batch: Iterable[Dict[str, np.ndarray]], pad_value: int = 0):
    output = {key: [] for key in batch[0]}
    output["mask"] = []
    max_size = 0

    # collate items to a list and find max number of detection in this batch
    for item in batch:
        for key, value in item.items():
            output[key].append(value)
        max_size = max(max_size, len(item["labels"]))
    
    # pad labels and masks to max length
    for i in range(len(batch)):
        item_size = len(output["labels"][i])
        output["mask"].append([1]*item_size)

        for _ in range(max_size - item_size):
            output["bboxes"][i].append([pad_value]*4)
            output["labels"][i].append(pad_value)
            output["mask"][i].append(0)    
    
    # image is a list of tensor -> use torch.stack
    # the rest are nested lists -> use torch.tensor
    for key, value in output.items():
        if key != "image":
            output[key] = torch.tensor(value)
        else:
            output[key] = torch.stack(value, dim=0)

    return output

class InferenceDataset(Dataset):
    """Dataset used for inference. Each item is a dict with keys `image`, `original_height`, and `original_width`.
    """
    def __init__(self, data_dir: str, resize_height: int = 512, resize_width: int = 512):
        assert os.path.exists(data_dir), f"{data_dir} does not exist"

        supported_formats = ["bmp", "jpeg", "jpg", "png"]   # cv2 is used to read the image. add image extension here if you need to read other formats
        files = [x for x in os.listdir(data_dir) if x.split(".")[-1].lower() in supported_formats]

        transforms = A.Compose([
            A.Resize(height=resize_height, width=resize_width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255),
            ToTensorV2()
        ])

        self.data_dir = data_dir
        self.img_names = files
        self.transforms = transforms

    def __getitem__(self, index: int):
        img = os.path.join(self.data_dir, self.img_names[index])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width, _ = img.shape
        img = self.transforms(image=img)["image"]

        item = {
            "image": img,
            "original_height": height,
            "original_width": width
        }
        return item
    
    def __len__(self):
        return len(self.img_names)
