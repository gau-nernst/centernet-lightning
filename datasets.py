from albumentations.core.serialization import save
import cv2
import os
import pickle
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

def prepare_coco_detection(data_dir, coco_name):
    ann_dir = os.path.join(data_dir, "annotations")
    ann_file = os.path.join(ann_dir, f"instances_{coco_name}.json")

    save_dir = os.path.join(ann_dir, coco_name)
    os.makedirs(save_dir, exist_ok=True)
    detection_file = os.path.join(save_dir, "detections.pkl")
    label_map_file = os.path.join(save_dir, "label_map.pkl")

    # if already exist on disk, don't do anything
    if os.path.exists(detection_file) and os.path.exists(label_map_file):
        return

    # extract only bboxes and ids data, otherwise train set annotations is too large
    else:
        coco = COCO(ann_file)

        # since COCO 2017 have missing ids, re-map the ids to class labels
        categories = OrderedDict(coco.cats)
        id_to_label = {}
        label_to_name = {}
        for i,(k,v) in enumerate(categories.items()):
            id_to_label[k] = i
            label_to_name[i] = v["name"]

        with open(label_map_file, "wb") as f:
            pickle.dump(label_to_name, f)     # save to disk

        img_ids = coco.getImgIds()                          # list of all image ids
        img_info = coco.loadImgs(img_ids)                   # list of dictionary
        img_names = [x["file_name"] for x in img_info]      # we only need file_name
        img_widths = [x["width"] for x in img_info]
        img_heights = [x["height"] for x in img_info]

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

        # bboxes = [[x["bbox"] for x in ann] for ann in annotates]        # we only need bboxes and category_id
        # labels = [[id_to_label[x["category_id"]] for x in ann] for ann in annotates]
        
        detection = {
            "img_ids": img_ids,
            "img_names": img_names,
            "img_widths": img_widths,
            "img_heights": img_heights,
            "bboxes": bboxes,
            "labels": labels
        }
        with open(detection_file, "wb") as f:
            pickle.dump(detection, f)          # save to disk

        detection["label_to_name"] = label_to_name

        del coco
    

class COCODataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        data_name: str, 
        transforms=None, 
        img_width: int=512, 
        img_height: int=512,
        eval: bool=False
        ):
        super(COCODataset, self).__init__()

        detection_file = os.path.join(data_dir, "annotations", data_name, "detections.pkl")
        label_map_file = os.path.join(data_dir, "annotations", data_name, "label_map.pkl")
        with open(detection_file, "rb") as f:
            detection = pickle.load(f)

        with open(label_map_file, "rb") as f:
            label_to_name = pickle.load(f)

        self.eval = eval
        if eval:
            self.img_ids = detection["img_ids"]
            self.original_widths = detection["img_widths"]
            self.original_heights = detection["img_heights"]
        
        self.img_names = detection["img_names"]
        self.bboxes = detection["bboxes"]
        self.labels = detection["labels"]
        self.label_to_name = label_to_name
        self.num_classes = len(label_to_name)
        self.img_dir = os.path.join(data_dir, data_name)

        # default transforms is convert to tensor
        if transforms == None:
            # use Albumentation resize to handle bbox resizing also
            # centernet resize input to 512 and 512
            # yolo bbox format is cxcywh
            transforms = A.Compose([
                A.Resize(img_height, img_width),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=["labels"]))
            
        self.transforms = transforms
        self.img_width = img_width
        self.img_height = img_height


    def __getitem__(self, index: int):
        # coco data format https://cocodataset.org/#format-data
        img_path = os.path.join(self.img_dir, self.img_names[index])

        # read image with cv2. convert to rgb color
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes = self.bboxes[index]
        labels = self.labels[index]

        # self.transforms is an Albumentations Transform instance
        # Albumentations will handle transforming the bounding boxes also
        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
        img = augmented["image"] / 255.     # don't know why Albumentations doesn't divide by 255 by default
        bboxes = augmented["bboxes"]
        labels = augmented["labels"]

        for i in range(len(bboxes)):
            x,y,w,h = bboxes[i]
            # clip boxes
            # x = max(x,0)
            # y = max(y,0)
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            w = min(w, self.img_width-x)
            h = min(h, self.img_height-y)

            # convert COCO xywh to cxcywh and convert to relative scale
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
        if self.eval:
            data["img_ids"] = self.img_ids[index]
            data["original_widths"] = self.original_widths[index]
            data["original_heights"] = self.original_heights[index]
        
        return data

    def __len__(self):
        return len(self.img_names)

def collate_detections_with_padding(batch, pad_value=0):
    output = {key: [] for key in batch[0]}
    output["mask"] = []
    max_size = 0

    for item in batch:
        # output["image"].append(item["image"])
        # output["bboxes"].append(item["bboxes"])
        # output["labels"].append(item["labels"])
        for key in item:
            output[key].append(item[key])

        max_size = max(max_size, len(item["labels"]))
    
    for i in range(len(batch)):
        item_size = len(output["labels"][i])
        output["mask"].append([1]*item_size)

        for _ in range(max_size - item_size):
            output["bboxes"][i].append([pad_value]*4)
            output["labels"][i].append(pad_value)
            output["mask"][i].append(0)    
    
    output["image"] = torch.stack(output["image"], dim=0)
    # output["bboxes"] = torch.tensor(output["bboxes"])
    # output["labels"] = torch.tensor(output["labels"])
    # output["mask"] = torch.tensor(output["mask"])
    for key, value in output.items():
        if key != "image":
            output[key] = torch.tensor(value)
    
    return output
