import numpy as np
from torchvision import datasets
import albumentations as A


class CocoDetection(datasets.CocoDetection):
    def __init__(self, img_dir, ann_json, transforms: A.Compose=None):
        if transforms is not None:
            box_params = transforms.processors["bboxes"].params
            assert box_params.format == "coco"
            assert "labels" in box_params.label_fields
        
        super().__init__(img_dir, ann_json)
        self._transforms = transforms

        ann_keys = ("bbox", "category_id")
        self.coco.anns = {ann_id: {k: ann[k] for k in ann_keys} for ann_id, ann in self.coco.anns.items()}
    
    def _load_target(self, idx):
        target = super()._load_target(idx)
        return {
            "boxes": [x["bbox"] for x in target],
            "labels": [x["category_id"] for x in target]
        }

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = np.array(img)
        img_h, img_w = img.shape[:2]
        target["image_width"] = img_w
        target["image_height"] = img_h

        if self._transforms is not None:
            augmented = self._transforms(image=img, bboxes=target["boxes"], labels=target["labels"])
            img = augmented["image"]
            target["boxes"] = augmented["bboxes"]
            target["labels"] = augmented["labels"]

        return img, target
