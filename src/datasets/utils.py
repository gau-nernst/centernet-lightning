from typing import Dict, Iterable
import math

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from numba import njit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_default_transforms(format="yolo"):
    transforms = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255),
        A.Resize(512, 512),
        ToTensorV2()
    ]
    # bbox_params = A.BboxParams(format=format, label_fields=["labels"], min_area=1024, min_visibility=0.1)
    bbox_params = A.BboxParams(format=format, label_fields=["labels"])

    transforms = A.Compose(transforms, bbox_params=bbox_params)
    return transforms

class CollateDetection:
    # def __init__(self, num_classes, heatmap_height, heatmap_width):
    #     self.num_classes = num_classes
    #     self.heatmap_

    def __call__(self, batch: Iterable):
        """Receive a batch of items, each contains the following keys:
            - image: torch.Tensor image in CHW format
            - bboxes: nested list of bounding boxes
            - labels: a list of labels
        """
        batch_size = len(batch)
        max_length = max(len(x["labels"]) for x in batch)

        bboxes = np.zeros(shape=(batch_size, max_length, 4), dtype=np.float32)
        labels = np.zeros(shape=(batch_size, max_length), dtype=np.int32)
        mask = np.zeros(shape=(batch_size, max_length), dtype=np.uint8)
        # heatmap = np.zeros(shape=(batch_size, self.num_classes, self.heatmap_height, self.heatmap_width), dtype=np.float32)
        
        for b, item in enumerate(batch):
            num_detections = len(item["labels"])
            if num_detections > 0:
                bboxes[b,:num_detections] = item["bboxes"]
                labels[b,:num_detections] = item["labels"]
                mask[b,:num_detections] = 1

        image = torch.stack([x["image"] for x in batch], dim=0)   # NCHW
        bboxes = torch.from_numpy(bboxes)
        labels = torch.from_numpy(labels)
        mask = torch.from_numpy(mask)

        output = {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "mask": mask
        }
        return output

def render_target_heatmap_ttfnet(
    heatmap: np.ndarray,
    bboxes: Iterable[Iterable],
    labels: Iterable, 
    alpha: float = 0.54,
    eps: float = 1e-8
    ):
    """Render target heatmap for 1 image. TTFNet method. Reference implementation https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L241

    Args
        heatmap: np.ndarray with shape CHW e.g. 80 x 128 x 128
        bboxes: a list of bounding boxes in cxcywh format
        labels: a list of labels [0, num_classes-1]
        alpha: parameter to calculate Gaussian std
    """
    _, img_height, img_width = heatmap.shape

    # a matrix of (x,y)
    grid_y = np.arange(img_height).reshape(-1,1)
    grid_x = np.arange(img_width).reshape(1,-1)

    for box, label in zip(bboxes, labels):
        cx, cy, w, h = box
        cx = int(cx * img_width)
        cy = int(cy * img_height)
        w *= img_width
        h *= img_height

        # From TTFNet
        var_w = np.square(alpha * w / 6)
        var_h = np.square(alpha * h / 6)

        # gaussian kernel
        radius_sq = np.square(cx - grid_x) / (2*var_w + eps) + np.square(cy - grid_y) / (2*var_h + eps)
        gaussian = np.exp(-radius_sq)
        np.maximum(heatmap[label], gaussian, out=heatmap[label])

    return heatmap

def render_target_heatmap_cornernet(
    heatmap: np.ndarray,
    bboxes: Iterable[Iterable],
    labels: Iterable,
    min_overlap: float = 0.7,
    eps: float = 1e-8
    ):
    """Render target heatmap for 1 image. CornerNet method. Reference implementation https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py

    Args
        heatmap: np.ndarray with shape CHW e.g. 80 x 128 x 128
        bboxes: a list of bounding boxes in cxcywh format
        labels: a list of labels [0, num_classes-1]
        min_overlap: parameter to calculate Gaussian radius
    """
    _, img_height, img_width = heatmap.shape

    for box, label in zip(bboxes, labels):
        cx, cy, w, h = box
        cx = int(cx * img_width)
        cy = int(cy * img_height)
        w *= img_width
        h *= img_height

        # calculate gaussian radius and clamp to 0
        radius = cornernet_gaussian_radius(w, h, min_overlap=min_overlap)
        radius = max(radius, 0)

        # calculate gaussian variance
        diameter = 2 * radius + 1
        var = np.square(diameter / 6)
        r = int(radius)

        # grid for the gaussian
        grid_y = np.arange(-r, r+1).reshape(-1,1)
        grid_x = np.arange(-r, r+1).reshape(1,-1)

        # generate the gaussian and clamp it
        gaussian = np.exp(-(grid_x*grid_x + grid_y*grid_y) / (2*var + eps))
        gaussian[gaussian < np.finfo(gaussian.dtype).eps * np.max(gaussian)] = 0

        # copy the gaussian over the output heatmap
        left   = min(cx, r)
        right  = min(img_width - cx, r + 1)
        top    = min(cy, r)
        bottom = min(img_height - cy, r + 1)

        masked_heatmap = heatmap[label, cy - top:cy + bottom, cx - left:cx + right]
        masked_gaussian = gaussian[r - top:r + bottom, r - left:r + right]
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap

@njit
def cornernet_gaussian_radius(width: float, height: float, min_overlap: float = 0.7):
    """Get radius for the Gaussian kernel. From CornerNet

    This is the bug-fixed version from CornerNet. Note that CenterNet used the bugged version
    https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    """
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 * b1 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 * b2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 * b3 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)
