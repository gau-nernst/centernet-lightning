import os
import json

from torch.utils.data import Dataset
import cv2

class CrowdHumanDataset(Dataset):
    def __init__(self, data_dir, split, transforms=None, name_to_label={"person": 0, "mask": 1}, img_ext=".jpg"):
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
            img_labels = []
            img_bboxes = []
            for detection in line["gtboxes"]:
                label = name_to_label[detection["tag"]]
                
                x, y, w, h = detection["fbox"]
                if w < 1 or h < 1:
                    continue
                # since image width and height are not present in annotations
                # we can only clip x2 y2 in __getitem__()
                x2 = x + w
                y2 = y + h
                x1 = max(0, x)
                y1 = max(0, y)
                box = [x1, y1, x2, y2]

                img_labels.append(label)
                img_bboxes.append(box)
            
            self.img_names.append(img_name)
            self.labels.append(img_labels)
            self.bboxes.append(img_bboxes)

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, self.img_names[index])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = img.shape

        labels = self.labels[index]
        bboxes = self.bboxes[index]

        for box in bboxes:
            box[2] = min(img_width-1, box[2])
            box[3] = min(img_height-1, box[3])

            box[2] = (box[2] - box[0]) / img_width      # w = x2 - x1
            box[3] = (box[3] - box[1]) / img_height     # h = y2 - y1
            box[0] = box[0] / img_width + box[2] / 2    # cx = x1 + w/2
            box[1] = box[1] / img_height + box[3] / 2   # cy = y1 + h/2

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
