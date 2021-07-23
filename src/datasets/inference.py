from typing import Iterable
import os
import warnings

import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import IMAGENET_MEAN, IMAGENET_STD

class InferenceDataset(Dataset):
    _IMG_EXT = (".jpg", ".jpeg", ".JPG", ".JPEG")

    def __init__(self, data_dir: str, img_names: Iterable[str] = None, resize_height: int = 512, resize_width: int = 512):
        """Dataset used for inference. Each item is a dict with keys `image`, `original_height`, and `original_width`.
        """
        super().__init__()
        assert os.path.exists(data_dir)

        transforms = A.Compose([
            A.Resize(height=resize_height, width=resize_width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255),
            ToTensorV2()
        ])

        if img_names is None:
            warnings.warn("img_names is not provided. JPEG files will be auto-discovered inside data_dir")
            img_names = [x for x in os.listdir(data_dir) if x.endswith(self._IMG_EXT)]
            img_names.sort()
            warnings.warn(f"{len(img_names)} JPEG files discovered")

        self.data_dir   = data_dir
        self.img_names  = img_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width, _ = img.shape
        img = self.transforms(image=img)["image"]

        item = {
            "image_path": img_path,
            "image": img,
            "original_height": height,
            "original_width": width
        }
        return item
    
    def __len__(self):
        return len(self.img_names)
