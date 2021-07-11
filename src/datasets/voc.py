import warnings
from typing import Dict
import os
import xml.etree.ElementTree as ET

import cv2
from torch.utils.data import Dataset
import albumentations as A

def process_voc_xml(ann_file, original_bboxes=False):
    tree = ET.parse(ann_file)
    root = tree.getroot()

    names = []
    bboxes = []

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    for x in root.iter("object"):
        name = x.find("name").text
        names.append(name)
        
        # clamp the coordinates
        x1 = max(float(x.find("bndbox/xmin").text), 0)
        y1 = max(float(x.find("bndbox/ymin").text), 0)
        x2 = min(float(x.find("bndbox/xmax").text), img_width-1)
        y2 = min(float(x.find("bndbox/ymax").text), img_height-1)

        if original_bboxes:
            bboxes.append([x1, y1, x2, y2])
        
        else:
            # x1x2y1y2 to normalized cxcywh
            cx = (x1 + x2) / (2 * img_width)
            cy = (y1 + y2) / (2 * img_height)
            w  = (x2 - x1) / img_width
            h  = (y2 - y1) / img_height

            bboxes.append([cx, cy, w, h])

    item = {
        "img_width": img_width,
        "img_height": img_height,
        "names": names,
        "bboxes": bboxes
    }
    return item

class VOCDataset(Dataset):
    """Dataset class for data in PASCAL VOC format. Only detection is supported. Bounding box in YOLO format (cxcywh and normalized to [0,1])

    Args
        data_dir: root directory which contains ImageSets, Annotations, and JPEGImages
        split: the split to use, which must be a file inside `data_dir/ImageSets/Main`
        transforms (optional): albumentation transform
        name_to_label (optional): a dict to map label name (in text) to label number (or class id)
    """
    def __init__(self, data_dir: str, split: str, transforms: A.Compose = None, name_to_label: Dict = None):
        super().__init__()
        if name_to_label is None:
            warnings.warn("name_to_label dict is not provided. label will be the original text name in the annotation file")

        self.img_dir = os.path.join(data_dir, "JPEGImages")
        self.transforms = transforms

        img_list = os.path.join(data_dir, "ImageSets", "Main", f"{split}.txt")      # VOC/ImageSets/Main/train.txt
        with open(img_list, "r") as f:
            self.img_names = [x.rstrip() for x in f]
        
        self.labels = []
        self.bboxes = []
        ann_dir = os.path.join(data_dir, "Annotations")
        for img_name in self.img_names:
            ann_file = os.path.join(ann_dir, f"{img_name}.xml")
            annotation = process_voc_xml(ann_file)

            img_labels = annotation["names"]
            img_bboxes = annotation["bboxes"]
            if name_to_label is not None:
                img_labels = [name_to_label[x] for x in img_labels]

            self.labels.append(img_labels)
            self.bboxes.append(img_bboxes)
        
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_name = os.path.join(self.img_dir, f"{img_name}.jpg")
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = self.bboxes[index]
        labels = self.labels[index]

        if self.transforms is not None:        
            augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
            img = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["labels"]

        item = {
            "image": img,
            "bboxes": bboxes,
            "labels": labels
        }
        return item

    def __len__(self):
        return len(self.img_names)
