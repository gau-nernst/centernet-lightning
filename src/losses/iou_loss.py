import torch
from torch import nn

__all__ = ["CenterNetIoULoss", "CenterNetGIoULoss"]

class CenterNetIoULoss(nn.Module):
    def __init__(self, reduction="none", keepdim=True):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.reduction = reduction
        self.keepdim = keepdim      # keepdim is to comply dimension with L1 loss

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-8):
        """inputs and targets are in (w,h) format
        """
        w1 = boxes1[...,0]
        h1 = boxes1[...,1]
        w2 = boxes2[...,0]
        h2 = boxes2[...,1]

        intersection = torch.minimum(w1, w2) * torch.minimum(h1, h2)    # I = min(w1, w2) * min(h1, h2)
        union = w1*h1 + w2*h2 - intersection                            # U = w1h1 + w2h2 - I
        
        iou = intersection / (union + eps)
        loss = 1 - iou

        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.keepdim:
            return loss.unsqueeze(-1)
        return loss

class CenterNetGIoULoss(nn.Module):
    def __init__(self, reduction="none", keepdim=True):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.reduction = reduction
        self.keepdim = keepdim      # keepdim is to comply dimension with L1 loss

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-8):
        w1 = boxes1[...,0]
        h1 = boxes1[...,1]
        w2 = boxes2[...,0]
        h2 = boxes2[...,1]

        intersection = torch.minimum(w1, w2) * torch.minimum(h1, h2)    # I = min(w1, w2) * min(h1, h2)
        union = w1*h1 + w2*h2 - intersection                            # U = w1h1 + w2h2 - I
        enclosed = torch.maximum(w1, w2) * torch.maximum(h1, h2)        # C = max(w1, w2) * max(h1, h2)

        giou = intersection / (union + eps) - (enclosed - union) / (enclosed + eps)
        loss = 1 - giou

        if self.reduction == "sum":
            return torch.sum(loss)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.keepdim:
            return loss.unsqueeze(-1)
        return loss
