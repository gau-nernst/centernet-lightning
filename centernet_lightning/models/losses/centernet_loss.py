from typing import Callable, List, Tuple, Iterable, Dict, Any
import math
import itertools

import torch
from torch import nn

from ..centernet import CenterNetDecoder

_Box = Tuple[float, float, float, float]
_two_int = Tuple[int, int]


def ttfnet_radius(w: float, h: float, alpha=0.54):
    return w / 2 * alpha, h / 2 * alpha


# Explanation: https://github.com/princeton-vl/CornerNet/issues/110
# Source: https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
def cornernet_radius(w: float, h: float, min_overlap: float = 0.3):
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    r = min(r1, r2, r3)
    return r, r


class CenterNetLoss(nn.Module):
    def __init__(self, heatmap_loss: nn.Module, box_loss: nn.Module, radius_function: Callable[[float, float], Tuple[float, float]]):
        super().__init__()
        self.heatmap_loss = heatmap_loss
        self.box_loss = box_loss
        self.radius_function = radius_function
        self.decoder = CenterNetDecoder()

    def calculate_centers_radii(self, boxes: Iterable[_Box], stride: int = 4) -> Tuple[List[int], List[int]]:
        # centers and radii are in heatmap coordinates
        centers_int = []
        radii_int = []
        for box in boxes:
            x, y, w, h = [d / stride for d in box]
            cx = round(x + w / 2)
            cy = round(y + h / 2)
            centers_int.append((cx, cy))
        
            # TODO: check CenterNet, mmdet implementation, and CenterNet2
            rx, ry = self.radius_function(w, h)
            rx = max(0, round(rx))
            ry = max(0, round(ry))
            radii_int.append((rx, ry))
        return centers_int, radii_int

    # https://github.com/princeton-vl/CornerNet/blob/master/sample/coco.py
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    @staticmethod
    def update_heatmap(
        heatmap: torch.Tensor,
        centers_int: Iterable[_two_int],
        radii_int: Iterable[_two_int],
        labels: Iterable[int]
    ) -> torch.Tensor:
        out_h, out_w = heatmap.shape[-2:]

        # TODO: check CenterNet, mmdet implementation, and CenterNet2
        for (cx, cy), (rx, ry), label in zip(centers_int, radii_int, labels):
            # only gaussian and heatmap are on gpu
            std_x = rx/3 + 1/6 
            std_y = ry/3 + 1/6
            grid_x = torch.arange(-rx, rx+1, device=heatmap.device).view(1,-1)
            grid_y = torch.arange(-ry, ry+1, device=heatmap.device).view(-1,1)

            gaussian = grid_x.square() / (2 * std_x**2) + grid_y.square() / (2 * std_y**2)
            gaussian = torch.exp(-gaussian)
            gaussian[gaussian < torch.finfo(gaussian.dtype).eps * torch.max(gaussian)] = 0

            l = min(cx, rx)
            t = min(cy, ry)
            r = min(out_w - cx, rx + 1)
            b = min(out_h - cy, ry + 1)
            masked_heatmap = heatmap[label, cy - t : cy + b, cx - l : cx + r]
            masked_gaussian = gaussian[ry - t : ry + b, rx - l : rx + r]
            torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
        
        return heatmap
    
    @staticmethod
    def sample_nxn_boxes(
        boxes: Iterable[_Box],
        centers_int: Iterable[_two_int],
        img_w: int,
        img_h: int,
        n: int = 3
    ) -> Tuple[List[_Box], List[int]]:
        # boxes are in input image coordinates
        # centers_int are in output heatmap coordinates
        box_samples, indices = [], []
        for (x, y, w, h), (cx, cy) in zip(boxes, centers_int):
            xyxy_box = (x, y, x + w, y + h)
            
            start_x = max(0, cx - n // 2)
            end_x = min(img_w - 1, cx + n // 2)
            cxs = list(range(start_x, end_x + 1))

            start_y = max(0, cy - n // 2)
            end_y = min(img_h - 1, cy + n // 2)
            cys = list(range(start_y, end_y + 1))

            new_centers = itertools.product(cxs, cys)
            for cx, cy in new_centers:
                indices.append(cy * img_w + cx)
                box_samples.append(xyxy_box)
        return box_samples, indices

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        stride: int = 4
    ):
        heatmap, box_offsets = outputs["heatmap"], outputs["box_2d"]
        target_boxes, target_labels = targets["boxes"], targets["label"]
        batch_size, _, out_h, out_w = heatmap.shape
        dtype, device = heatmap.dtype, heatmap.device

        target_heatmap = torch.zeros_like(heatmap)
        box_loss = torch.tensor(0., dtype=dtype, device=device)
        for i, (boxes, labels) in enumerate(zip(target_boxes, target_labels)):
            if len(boxes) == 0:
                continue

            centers_int, radii_int = self.calculate_centers_radii(boxes, stride=stride)
            self.update_heatmap(target_heatmap[i], centers_int, radii_int, labels)

            box_samples, indices = self.sample_nxn_boxes(boxes, centers_int, out_w, out_h)
            pred_boxes = self.decoder.gather_and_decode_boxes(
                box_offsets[i],
                torch.tensor(indices, device=device)
            )
            box_loss = box_loss + self.box_loss(pred_boxes, torch.tensor(box_samples, device=device))

        heatmap_loss = self.heatmap_loss(heatmap, target_heatmap)
        box_loss = box_loss / batch_size

        return {'heatmap': heatmap_loss, 'box_2d': box_loss}
