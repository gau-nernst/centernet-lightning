import contextlib
import logging
import os
from typing import Dict, Iterable, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


_Box = Tuple[float, float, float, float]


class _LogWriter:
    def write(self, msg: str) -> None:
        for line in msg.rstrip().splitlines():
            LOGGER.info(line.rstrip())

    def flush(self) -> None:
        pass


def _clip_box(xywh_box: Iterable[float], img_w: int, img_h: int) -> _Box:
    max_x = img_w - 1
    max_y = img_h - 1
    x1, y1, w, h = xywh_box
    x2, y2 = x1 + w, y1 + h
    x1, x2 = max(0, x1), min(max_x, x2)
    y1, y2 = max(0, y1), min(max_y, y2)
    return (x1, y1, x2 - x1, y2 - y1)


# references
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/coco.py
# https://cocodataset.org/#format-data
class CocoDetectionDataset(Dataset):
    def __init__(
        self, img_dir: str, ann_json: str, transforms: Optional[A.Compose] = None
    ):
        super().__init__()

        if transforms is not None:
            box_params = transforms.processors["bboxes"].params
            assert box_params.format == "coco"
            assert "labels" in box_params.label_fields

        LOGGER.info(f"Reading COCO annotation {ann_json}")
        with contextlib.redirect_stdout(_LogWriter()):  # redict pycocotools print()
            coco = COCO(ann_json)

        cat_ids = sorted(coco.getCatIds())
        label_to_id = {label: idx for idx, label in enumerate(cat_ids)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        img_ids = sorted(coco.getImgIds())

        data = []
        num_removed = 0
        for img_id in tqdm(img_ids, desc="Preprocess detections"):
            img_info = coco.imgs[img_id]  # keys: bbox, category_id, id
            img_anns = coco.imgToAnns[img_id]  # keys: filename, height, width, id

            boxes, labels = [], []
            for ann in img_anns:
                box = _clip_box(ann["bbox"], img_info["width"], img_info["height"])
                if min(box[2:]) < 1:  # remove degenerate boxes
                    num_removed += 1
                    continue
                boxes.append(box)
                labels.append(label_to_id[ann["category_id"]])

            img_data = {
                "filename": img_info["file_name"],
                "bboxes": np.array(boxes),
                "labels": np.array(labels),
                "img_width": img_info["width"],
                "img_height": img_info["height"],
                "img_id": img_id,
            }
            data.append(img_data)

        if num_removed > 0:
            LOGGER.warning(f"Removed {num_removed} empty box(es)")

        self.img_dir = img_dir
        self.data = data
        self.transforms = transforms
        self.num_classes = len(cat_ids)
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, np.ndarray]]:
        img_data = self.data[idx]
        img_path = os.path.join(self.img_dir, img_data["filename"])
        img = np.array(Image.open(img_path).convert("RGB"))

        boxes, labels = img_data["bboxes"], img_data["labels"]
        if self.transforms is not None:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = np.array(augmented["bboxes"])
            labels = np.array(augmented["labels"])

        target = {"bboxes": boxes, "labels": labels}
        return img, target

    def __len__(self) -> int:
        return len(self.data)


def coco_detection_collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    targets = tuple(x[1] for x in batch)
    return images, targets


# def parse_albumentations_transforms(transforms, box_params=None):
#     ts = []
#     for t in transforms:
#         t_fn = _transforms[t['name']]
#         init_args = t['init_args'] if 'init_args' in t else {}
#         ts.append(t_fn(**init_args))
#     ts.append(ToTensorV2())

#     if box_params is None:
#         box_params = {'format': 'coco', 'label_fields': ['labels'], 'min_area': 1}
#     return A.Compose(ts, bbox_params=box_params)
