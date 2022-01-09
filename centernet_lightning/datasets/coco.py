import os
import contextlib
import io

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image


def _clip_box(xywh_box, img_w, img_h):
    max_x = img_w - 1
    max_y = img_h - 1
    x1, y1, w, h = xywh_box
    x2, y2 = x1 + w, y1 + h
    x1, x2 = max(0, x1), min(max_x, x2)
    y1, y2 = max(0, y1), min(max_y, y2)
    return (x1, y1, x2-x1, y2-y1)


class CocoDetection(Dataset):
    def __init__(self, img_dir, ann_json, transforms: A.Compose=None):
        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/coco.py
        # https://cocodataset.org/#format-data
        if transforms is not None:
            box_params = transforms.processors["bboxes"].params
            assert box_params.format == "coco"
            assert "labels" in box_params.label_fields
        
        with contextlib.redirect_stdout(io.StringIO()):     # redict pycocotools print()
            coco = COCO(ann_json)
        cat_ids = sorted(coco.getCatIds())
        label_map = {v: i for i, v in enumerate(cat_ids)}
        inverse_label_map = {v: k for k, v in label_map.items()}

        img_ids = sorted(coco.getImgIds())
        imgs = coco.loadImgs(img_ids)                       # each img has keys filename, height, width, id
        target = [coco.imgToAnns[idx] for idx in img_ids]   # each ann has keys bbox, category_id, id
        
        img_names = [x["file_name"] for x in imgs]
        targets = [{
            "boxes": [ann["bbox"] for ann in img_anns],
            "labels": [label_map[ann["category_id"]] for ann in img_anns],
            "image_width": img["width"],
            "image_height": img["height"],
            "image_id": img["id"]
        } for img_anns, img in zip(target, imgs)]

        for target in targets:
            # clip boxes
            target["boxes"] = [_clip_box(box, target["image_width"], target["image_height"]) for box in target["boxes"]]
            
            # remove empty boxes
            boxes, labels = [], []
            for box, label in zip(target["boxes"], target["labels"]):
                if min(box[2:]) > 1:
                    boxes.append(box)
                    labels.append(label)            
            target["boxes"] = boxes
            target["labels"] = labels

        self.img_dir = img_dir
        self.img_names = img_names
        self.targets = targets
        self.transforms = transforms
        self.num_classes = len(cat_ids)
        self.label_map = label_map
        self.inverse_label_map = inverse_label_map

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = np.array(Image.open(img_path).convert("RGB"))
        
        target = self.targets[idx]
        assert target["image_width"] == img.shape[1]
        assert target["image_height"] == img.shape[0]

        if self.transforms is not None:
            augmented = self.transforms(image=img, bboxes=target["boxes"], labels=target["labels"])
            img = augmented["image"]
            target["boxes"] = augmented["bboxes"]
            target["labels"] = augmented["labels"]

        return img, target

    def __len__(self):
        return len(self.img_names)


def collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    targets = tuple(x[1] for x in batch)
    return images, targets
