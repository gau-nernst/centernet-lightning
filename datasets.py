import cv2
import os
import copy
from pycocotools.coco import COCO
from collections import OrderedDict

from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO: convert ground truth to appropriate output

class CenterNetDataset(Dataset):
    def __init__(self, img_dir: str) -> None:
        super(CenterNetDataset, self).__init__()
        self.img_dir = img_dir
        self.imgs = os.listdir(img_dir)
        self.imgs.sort()

    def __getitem__(self, idx: int):
        img = os.path.join(self.img_dir, self.imgs[idx])
        img = torchvision.io.read_image(img).float() / 255.

        item = {
            "image": img,
            "size": None,
            "offset": None,
            "displacement": None
        }
        return item

    def __len__(self):
        return len(self.img_dir)

class COCODataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        data_name: str, 
        transforms=None, 
        img_width: int=512, 
        img_height: int=512
        ):
        super(COCODataset, self).__init__()

        # COCO stuff
        ann_file = os.path.join(data_dir, "annotations", f"instances_{data_name}.json")
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.img_dir = os.path.join(data_dir, data_name)
        # use ordered dict to make it reproducible
        ordered_cat = OrderedDict(self.coco.cats)
        self.id_to_label = {k:i for i,k in enumerate(ordered_cat)}
        self.label_to_name = {i:v["name"] for i,v in enumerate(ordered_cat.values())}
        self.num_classes = len(self.id_to_label)

        # default transforms is convert to tensor
        if transforms == None:
            # use Albumentation resize to handle bbox resizing also
            # centernet resize input to 512 and 512
            # yolo bbox format is center xy wh
            transforms = A.Compose([
                A.Resize(img_height, img_width),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=["labels"]))
            
        self.transforms = transforms
        self.img_width = img_width
        self.img_height = img_height


    def __getitem__(self, idx: int):
        # coco data format https://cocodataset.org/#format-data
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        # img_info has the following keys: license, file_name, coco_url, height, width, date_captured, flickr_url, id
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # read image with cv2. convert to rgb color
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))
        # annotations is a list of annotataion
        # each annotation is a dictionary with keys: segmentation, area, iscrowd, image_id, bbox, category_id, id 

        bboxes = [x["bbox"] for x in anns]
        labels = [self.id_to_label[x["category_id"]] for x in anns]

        # self.transforms is an Albumentations Transform instance
        # Albumentations will handle transforming the bounding boxes also
        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
        img = augmented["image"] / 255.     # don't know why Albumentations doesn't divide by 255 by default
        bboxes = augmented["bboxes"]
        labels = augmented["labels"]

        for i in range(len(bboxes)):
            x,y,w,h = bboxes[i]
            # clip boxes
            x = max(x,0)
            y = max(y,0)
            w = min(w, self.img_width-x)
            h = min(h, self.img_height-y)

            # convert COCO xywh to cxcywh and change to relative scale
            center_x = (x + w/2) / self.img_width
            center_y = (y + h/2) / self.img_height
            w = w / self.img_width
            h = h / self.img_height

            bboxes[i] = [center_x, center_y, w, h]

        data = {
            "image": img,
            "bboxes": bboxes,
            "labels": labels
        }
        return data

    def __len__(self):
        return len(self.img_ids)

def collate_bbox_labels(batch, pad_value=0):
    output = {
        "image": [],
        "bboxes": [],
        "labels": [],
        "mask": []
    }
    max_size = 0

    for item in batch:
        output["image"].append(item["image"])
        output["bboxes"].append(item["bboxes"])
        output["labels"].append(item["labels"])

        max_size = max(max_size, len(item["labels"]))
    
    for i in range(len(batch)):
        item_size = len(output["labels"][i])
        output["mask"].append([1]*item_size)

        for _ in range(max_size - item_size):
            output["bboxes"][i].append([pad_value]*4)
            output["labels"][i].append(pad_value)
            output["mask"][i].append(0)    
    
    output["image"] = torch.stack(output["image"], dim=0)
    output["bboxes"] = torch.Tensor(output["bboxes"])
    output["labels"] = torch.Tensor(output["labels"])
    output["mask"] = torch.Tensor(output["mask"])
    
    return output
