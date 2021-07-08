import math

import torch
from torch import nn

def _get_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    area2 = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])
    
    x1_inter = torch.maximum(boxes1[...,0], boxes2[...,0])
    y1_inter = torch.maximum(boxes1[...,1], boxes2[...,1])
    x2_inter = torch.minimum(boxes1[...,2], boxes2[...,2])
    y2_inter = torch.minimum(boxes1[...,3], boxes2[...,3])

    intersection = torch.clamp_min(x2_inter - x1_inter, 0) * torch.clamp_min(y2_inter - y1_inter, 0)
    union = area1 + area2 - intersection

    return intersection, union

def _get_enclosing_box(boxes1: torch.Tensor, boxes2: torch.Tensor):
    x1_close = torch.minimum(boxes1[...,0], boxes2[...,0])
    y1_close = torch.minimum(boxes1[...,1], boxes2[...,1])
    x2_close = torch.maximum(boxes1[...,2], boxes2[...,2])
    y2_close = torch.maximum(boxes1[...,3], boxes2[...,3])

    return x1_close, y1_close, x2_close, y2_close

class IoULoss(nn.Module):
    def __init__(self, reduction="none", keepdim=True):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.reduction = reduction
        self.keepdim = keepdim      # keepdim is to comply dimension with L1 loss

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-8):
        """inputs and targets are in (w,h) format
        """
        intersection, union = _get_iou(boxes1, boxes2)
        iou = intersection / (union + eps)
        loss = 1 - iou

        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.keepdim:
            return loss.unsqueeze(-1)
        return loss

class GIoULoss(nn.Module):
    """Generalized IoU Loss. Paper: https://arxiv.org/abs/1902.09630
    """
    def __init__(self, reduction="none", keepdim=True):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.reduction = reduction
        self.keepdim = keepdim      # keepdim is to comply dimension with L1 loss

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-8):
        intersection, union = _get_iou(boxes1, boxes2)
        iou = intersection / (union + eps)

        x1_close, y1_close, x2_close, y2_close = _get_enclosing_box(boxes1, boxes2)
        enclosing = (x2_close - x1_close) * (y2_close - y1_close)
        giou = iou - (1 - union / enclosing)
        loss = 1 - giou

        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.keepdim:
            return loss.unsqueeze(-1)
        return loss

class DIoULoss(nn.Module):
    """Distance IoU Loss. Paper: https://arxiv.org/abs/1911.08287
    """
    def __init__(self, reduction="none", keepdim=True):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.reduction = reduction
        self.keepdim = keepdim      # keepdim is to comply dimension with L1 loss

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-8):
        intersection, union = _get_iou(boxes1, boxes2)
        iou = intersection / (union + eps)

        x1_close, y1_close, x2_close, y2_close = _get_enclosing_box(boxes1, boxes2)
        diagonal_sq = (x2_close - x1_close).square() + (y2_close - y1_close).square()
        centers1 = (boxes1[...,:2] + boxes1[...,2:]) / 2
        centers2 = (boxes2[...,:2] + boxes2[...,2:]) / 2
        distance_sq = (centers2[...,0] - centers1[...,0]).square() + (centers2[...,1] - centers1[...,1]).square()
        distance_penalty = distance_sq / diagonal_sq

        loss = 1 - iou + distance_penalty

        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.keepdim:
            return loss.unsqueeze(-1)
        return loss


class CIoULoss(nn.Module):
    """Complete IoU Loss. Paper: https://arxiv.org/abs/1911.08287
    """
    def __init__(self, reduction="none", keepdim=True):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.reduction = reduction
        self.keepdim = keepdim      # keepdim is to comply dimension with L1 loss

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-8):
        intersection, union = _get_iou(boxes1, boxes2)
        iou = intersection / (union + eps)

        # penalty for distance
        x1_close, y1_close, x2_close, y2_close = _get_enclosing_box(boxes1, boxes2)
        diagonal_sq = (x2_close - x1_close).square() + (y2_close - y1_close).square()
        centers1 = (boxes1[...,:2] + boxes1[...,2:]) / 2
        centers2 = (boxes2[...,:2] + boxes2[...,2:]) / 2
        distance_sq = (centers2[...,0] - centers1[...,0]).square() + (centers2[...,1] - centers1[...,1]).square()
        distance_penalty = distance_sq / diagonal_sq

        # penalty for aspect ratio
        w1 = boxes1[...,2] - boxes1[...,0]
        h1 = boxes1[...,3] - boxes1[...,1]
        w2 = boxes2[...,2] - boxes2[...,0]
        h2 = boxes2[...,3] - boxes2[...,1]
        
        angle_diff = (torch.atan(w1 / (h1+eps)) - torch.atan(w2 / (h2+eps))) * 2 / math.pi      # 2 / pi = 90 degree
        v = angle_diff.square()
        alpha = v / (1 - iou + v + eps)
        ratio_penalty = alpha * v

        loss = 1 - iou + distance_penalty + ratio_penalty

        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.keepdim:
            return loss.unsqueeze(-1)
        return loss
