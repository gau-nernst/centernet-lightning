from typing import Iterable
import os

import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import IMAGENET_MEAN, IMAGENET_STD

class InferenceDataset(Dataset):
    """Dataset used for inference. Each item is a dict with keys `image`, `original_height`, and `original_width`.
    """
    def __init__(self, data_dir: str, img_names: Iterable[str], resize_height: int = 512, resize_width: int = 512):
        transforms = A.Compose([
            A.Resize(height=resize_height, width=resize_width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255),
            ToTensorV2()
        ])

        self.data_dir   = data_dir
        self.img_names  = img_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        img = os.path.join(self.data_dir, self.img_names[index])
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width, _ = img.shape
        img = self.transforms(image=img)["image"]

        item = {
            "image": img,
            "original_height": height,
            "original_width": width
        }
        return item
    
    def __len__(self):
        return len(self.img_names)
