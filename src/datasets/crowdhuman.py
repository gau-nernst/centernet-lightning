import os
import json

from torch.utils.data import Dataset
import cv2
from PIL import Image

class CrowdHumanDataset(Dataset):
    def __init__(self, data_dir, split, transforms=None, name_to_label={"person": 0, "mask": 1}, ignore_mask=True, img_ext=".jpg"):
        super().__init__()
        self.img_dir = os.path.join(data_dir, split, "Images")
        self.transforms = transforms
        label_path = os.path.join(data_dir, split, f"annotation_{split}.odgt")

        annotations = []
        with open(label_path, "r") as f:
            for line in f:
                line = json.loads(line.rstrip())
                annotations.append(line)

        self.img_names = []
        self.labels = []
        self.bboxes = []
        for line in annotations:
            img_name = f"{line['ID']}{img_ext}"
            
            # annotations do not provide image dimensions
            # seems like PIL can read image dimensions without loading the image to memory
            img_path = os.path.join(self.img_dir, img_name)
            img = Image.open(img_path)
            img_width = img.width
            img_height = img.height
            
            img_labels = []
            img_bboxes = []
            for detection in line["gtboxes"]:
                if ignore_mask and detection["tag"] == "mask":
                    continue
                
                label = name_to_label[detection["tag"]]
                x, y, w, h = detection["fbox"]

                x2 = x + w
                y2 = y + h
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_width-1, x2)
                y2 = min(img_height-1, y2)

                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue

                cx = (x1 + x2) / 2 / img_width
                cy = (y1 + y2) / 2 / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height
                box = [cx,cy,w,h]

                img_labels.append(label)
                img_bboxes.append(box)
            
            self.img_names.append(img_name)
            self.labels.append(img_labels)
            self.bboxes.append(img_bboxes)

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, self.img_names[index])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = self.labels[index]
        bboxes = self.bboxes[index]

        if self.transforms is not None:
            augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
            return augmented

        item = {
            "image": img,
            "labels": labels,
            "bboxes": bboxes
        }
        return item

    def __len__(self):
        return len(self.labels)
