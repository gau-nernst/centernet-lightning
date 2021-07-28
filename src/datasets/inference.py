import os
import warnings

import cv2
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    """Dataset used for inference. Each item is a dict with keys `image`, `original_height`, and `original_width`.
    """
    def __init__(self, data_dir, img_names=None, transforms=None, file_ext=None): 
        super().__init__()
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self.transforms = transforms

        if img_names is None:
            warnings.warn("img_names is not provided. Files will be auto-discovered inside data_dir")
            img_names = os.listdir(data_dir)
            if file_ext is not None:
                img_names = [x for x in img_names if x.endswith(file_ext)]
            img_names.sort()
            warnings.warn(f"{len(img_names)} files discovered")

        self.img_names  = img_names
        
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

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
