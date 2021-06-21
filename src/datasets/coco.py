import warnings
from typing import Dict, Iterable
import os
from collections import OrderedDict
import json
import pickle

import cv2
from torch.utils.data import Dataset
import albumentations as A
from pycocotools.coco import COCO

from .utils import get_default_transforms

def prepare_coco_detection(ann_dir: str, split: str, overwrite: bool = False):
    ann_file = os.path.join(ann_dir, f"instances_{split}.json")
    det_file = os.path.join(ann_dir, f"detections_{split}.pkl")
    label_to_name_file = os.path.join(ann_dir, "label_to_name.json")
    label_to_id_file   = os.path.join(ann_dir, "label_to_id.json")

    # if already exist on disk, don't do anything
    if not overwrite and os.path.exists(det_file) and os.path.exists(label_to_name_file) and os.path.exists(label_to_id_file):
        return

    # extract only bboxes and ids data, otherwise train set annotations is too large
    coco = COCO(ann_file)
    categories = OrderedDict(coco.cats)
    
    label_to_id   = {i: v["id"] for i,v in enumerate(categories.values())}
    label_to_name = {i: v["name"] for i,v in enumerate(categories.values())}
    id_to_label   = {v: k for k,v in label_to_id.items()}

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
    with open(det_file, "wb") as f:
        pickle.dump(detection, f)          # save to disk

    del coco

def get_coco_subset(detection_file: str, indices: Iterable = None):
    if not indices:
        indices = range(16)
    elif isinstance(indices, int):
        indices = range(indices)
     
    with open(detection_file, "rb") as f:
        detection = pickle.load(f)
    
    sub_detection = {
        "img_names": [],
        "bboxes": [],
        "labels": []
    }
    for i in indices:
        for k in sub_detection.keys():
            sub_detection[k].append(detection[k][i])
    
    return sub_detection

class COCODataset(Dataset):
    """Dataset class for dataset in COCO format

    Args:
        data_dir: root directory, which contains folder `annotations` and `images`
        split: the split to use e.g. `train2017`. Annotation file e.g. `instances_train2017.json` must be present in the folder `annotations`. Image folder of the split e.g. `train2017` must be present in the folder `images`
        transforms: albumentation transform
    """
    def __init__(self, data_dir: str, split: str, transforms: A.Compose = None):
        super(COCODataset, self).__init__()
        ann_dir = os.path.join(data_dir, "annotataions")
        detection_file = os.path.join(ann_dir, "detections.pkl")

        # extract necessary info for detection
        if not os.path.exists(detection_file):
            prepare_coco_detection(ann_dir, split)

        with open(detection_file, "rb") as f:
            detection = pickle.load(f)

        if transforms is None:
            warnings.warn("transforms is not specified. Default to normalize with ImageNet and resize to 512x512")
            transforms = get_default_transforms(format="coco")

        self.img_dir = data_dir
        self.transforms = transforms

        self.img_names = detection["img_names"]
        self.bboxes    = detection["bboxes"]
        self.labels    = detection["labels"]

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

        # convert xywh to cxcywh
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
