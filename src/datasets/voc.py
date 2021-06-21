import warnings
from typing import Dict
import os
import xml.etree.ElementTree as ET

import cv2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import IMAGENET_MEAN, IMAGENET_STD, collate_detections_with_padding

class VOCDataset(Dataset):
    """Dataset class for data in PASCAL VOC format. Only detection is supported

    Args
        data_dir: root directory which contains ImageSets, Annotations, and JPEGImages
        split: the split to use, which must be a file inside `data_dir/ImageSets/Main`
        transforms (optional): albumentation transform
        name_to_label (optional): a dict to map label name (in text) to label number (or class id)
    """
    def __init__(self, data_dir: str, split: str, transforms: A.Compose = None, name_to_label: Dict = None):
        super().__init__()
        if transforms is None:
            warnings.warn("transforms is not specified. Default to normalize with ImageNet and resize to 512x512")
            # use Albumentation resize to handle bbox resizing also
            # centernet resize input to 512 and 512
            transforms = A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255),
                A.Resize(512, 512),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.1, label_fields=["labels"]))
        
        split_file = os.path.join(data_dir, "ImageSets", "Main", split)
        
        with open(split_file, "r") as f:
            img_names = [x.rstrip() for x in f]
        
        labels = []
        bboxes = []
        ann_dir = os.path.join(data_dir, "Annotations")
        for img_name in img_names:
            ann_file = os.path.join(ann_dir, img_name) + ".xml"
            tree = ET.parse(ann_file)
            root = tree.getroot()

            names = []
            boxes = []
            for x in root.iter("object"):
                name = x.find("name").text
                names.append(name)
                
                x1 = int(x.find("bndbox/xmin").text)
                y1 = int(x.find("bndbox/ymin").text)
                x2 = int(x.find("bndbox/xmax").text)
                y2 = int(x.find("bndbox/ymax").text)
                boxes.append([x1, y1, x2, y2])
                
            labels.append(names)
            bboxes.append(boxes)

        if name_to_label:
            labels = [[name_to_label[x] for x in img_labels] for img_labels in labels]

        self.img_names = img_names
        self.transforms = transforms
        self.img_dir = os.path.join(data_dir, "JPEGImages")
        self.labels = labels
        self.bboxes = bboxes

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_file = os.path.join(self.img_dir, img_name) + ".jpg"
        img = cv2.imread(img_file)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(img_name)

        bboxes = self.bboxes[index]
        labels = self.labels[index]

        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
        img = augmented["image"]
        bboxes = augmented["bboxes"]
        labels = augmented["labels"]

        # convert x1y1x2y2 to cxcywh
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            bboxes[i] = [cx, cy, w, h]

        item = {
            "image": img,
            "bboxes": bboxes,
            "labels": labels
        }
        return item

    def __len__(self):
        return len(self.img_names)
