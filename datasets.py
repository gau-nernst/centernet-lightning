import cv2
import os
import copy
from pycocotools.coco import COCO

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
# from torchvision.transforms import ColorJitter, RandomHorizontalFlip
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO: read detection data, COCO format
# TODO: add augmentations
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
            "img": img,
            "size": None,
            "offset": None,
            "displacement": None
        }
        return item

    def __len__(self):
        return len(self.img_dir)

class COCODataset(Dataset):
    def __init__(self, data_dir: str, data_name: str, transforms=None) -> None:
        super(COCODataset, self).__init__()

        ann_file = os.path.join(data_dir, "annotations", f"instances_{data_name}.json")
        self.coco = COCO(ann_file)
        self.imgs = self.coco.getImgIds()
        self.img_dir = os.path.join(data_dir, data_name)

        # default transforms is convert to tensor
        if transforms == None:
            transforms = A.Compose([
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco'))
            
        self.transforms = transforms


    def __getitem__(self, idx: int):
        img_id = self.imgs[idx]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # read image with cv2. convert to rgb color
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = copy.deepcopy(self.coco.loadAnns(ids=ann_ids))
        # annotations is a list of dictionary, each with keys ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        # COCO bbox is xywh

        # centers, sizes, heatmap

        bboxes = [x["bbox"] for x in anns]
        ids = [x["id"] for x in anns]

        # self.transforms is an Albumentations Transform instance
        # Albumentations will handle transforming the bounding boxes also
        augmented = self.transforms(image=img, bboxes=bboxes)
        img = augmented["image"]
        bboxes = augmented["bboxes"]

        # render gaussian heatmap

        data = {
            "img": img,
            "bboxes": bboxes,
            "ids": ids
        }
        return data

    def __len__(self):
        return len(self.imgs)

def render_gaussian_kernel(
    heatmap: torch.Tensor,
    center_x: float,
    center_y: float,
    box_w: float,
    box_h: float,
    alpha: float=0.54
    ):

    h, w = heatmap.shape
    dtype = heatmap.dtype
    device = heatmap.device

    # TTFNet
    std_w = alpha*box_w/6
    std_h = alpha*box_h/6
    var_w = std_w*std_w
    var_h = std_h*std_h

    # a matrix of (x,y)
    grid_y, grid_x = torch.meshgrid([
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device)]
    )

    radius_sq = (center_x - grid_x)**2/(2*var_w) + (center_y - grid_y)**2/(2*var_h)
    gaussian_kernel = torch.exp(-radius_sq)
    gaussian_kernel[center_y, center_x] = 1     # force the center to be 1
    heatmap = torch.maximum(heatmap, gaussian_kernel)
    return heatmap