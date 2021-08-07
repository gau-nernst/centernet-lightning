import os
from collections import OrderedDict
import json
import pickle

import cv2
from torch.utils.data import Dataset
import albumentations as A
from pycocotools.coco import COCO

def get_coco_detection_annotations(ann_dir: str, split: str, force_redo: bool = False):
    processed_ann = os.path.join(ann_dir, f"processed_detection_{split}.pkl")
    label_to_name_file = os.path.join(ann_dir, f"label_to_name_{split}.json")
    label_to_id_file   = os.path.join(ann_dir, f"label_to_id_{split}.json")

    if not force_redo and os.path.exists(processed_ann) and os.path.exists(label_to_name_file) and os.path.exists(label_to_id_file):
        with open(processed_ann, "rb") as f:
            ann = pickle.load(f)
        
        return ann

    # extract only bboxes and ids data, otherwise train set annotations is too large
    ann_file = os.path.join(ann_dir, f"instances_{split}.json")
    coco = COCO(ann_file)
    categories = OrderedDict(coco.cats)
    
    # label is used during training, with non-existent classes removed. id is the original labels 
    label_to_id   = {i: v["id"] for i,v in enumerate(categories.values())}
    label_to_name = {i: v["name"] for i,v in enumerate(categories.values())}
    id_to_label   = {v: k for k,v in label_to_id.items()}

    # save to disk
    with open(label_to_name_file, "w") as f:
        json.dump(label_to_name, f)
    with open(label_to_id_file, "w") as f:
        json.dump(label_to_id, f)

    img_ids = coco.getImgIds()                      # list of all image ids
    img_info = coco.loadImgs(img_ids)               # list of img info, each is a dict
    img_names = [x["file_name"] for x in img_info]  # we only need file_name to open the images
    img_widths = [x["width"] for x in img_info]     # to normalize boxes
    img_heights = [x["height"] for x in img_info]

    # get annotations for each image
    annotate_ids = [coco.getAnnIds(imgIds=x) for x in img_ids]
    annotates = [coco.loadAnns(ids=x) for x in annotate_ids]

    bboxes = []
    labels = []
    for img_detections, img_width, img_height in zip(annotates, img_widths, img_heights):
        img_bboxes = []
        img_labels = []
        for detection in img_detections:
            x1, y1, w, h = detection["bbox"]
            cat_id = detection["category_id"]

            # clip boxes
            x2 = x1 + w
            y2 = y1 + h
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width-1, x2)
            y2 = min(img_height-1, y2)

            # ignore boxes with width and height < 1. this will crash albumentation
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue

            # convert xywh to cxcywh and normalize to [0,1]
            cx = (x1 + x2) / 2 / img_width
            cy = (y1 + y2) / 2 / img_height
            w /= img_width
            h /= img_height

            img_bboxes.append([cx,cy,w,h])
            img_labels.append(id_to_label[cat_id])

        bboxes.append(img_bboxes)
        labels.append(img_labels)

    ann = {
        "img_ids": img_ids,
        "img_names": img_names,
        "bboxes": bboxes,
        "labels": labels
    }
    with open(processed_ann, "wb") as f:
        pickle.dump(ann, f)

    del coco
    return ann

class COCODataset(Dataset):
    """Dataset class for dataset in COCO format. Only detection is supported. Bounding box in YOLO format (cxcywh and normalized to [0,1])

    Args:
        data_dir: root directory, which contains folder `annotations` and `images`
        split: the split to use e.g. `train2017`. Annotation file e.g. `instances_train2017.json` must be present in the folder `annotations`. Image folder of the split e.g. `train2017` must be present in the folder `images`
        transforms: albumentation transform
    """
    def __init__(self, data_dir: str, split: str, transforms: A.Compose = None):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "images", split)
        self.transforms = transforms

        ann_dir = os.path.join(data_dir, "annotations")
        ann = get_coco_detection_annotations(ann_dir, split)
        self.img_names = ann["img_names"]
        self.bboxes = ann["bboxes"]
        self.labels = ann["labels"]

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_name = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = self.bboxes[index]
        labels = self.labels[index]

        # self.transforms is an Albumentations Transform instance
        # Albumentations will handle transforming the bounding boxes also
        if self.transforms is not None:
            augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
            return augmented
            
        item = {
            "image": img,
            "bboxes": bboxes,
            "labels": labels
        }
        return item

    def __len__(self):
        return len(self.img_names)
