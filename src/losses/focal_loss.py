import torch
from torch import nn
import torch.nn.functional as F

class ModifiedFocalLossWithLogits(nn.Module):
    """Implement Modified Focal Loss (from CornerNet) with Logits to improve numerical stability via logsigmoid. Default values are from the original paper.

    Args:
        alpha: control the modulating factor to reduce the impact of easy examples. This is gamma in the original Focal loss
        beta: control the additional weight for negative examples when y is between 0 and 1
        reduction: either none, sum, or mean 
    """
    # reference implementations
    # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gaussian_focal_loss.py
    def __init__(self, alpha: float = 2, beta: float = 4, reduction: str = "sum"):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # NOTE: targets is a 2D Gaussian
        pos_weight = targets.eq(1).float()              # gaussian peaks are positive samples
        neg_weight = torch.pow(1-targets, self.beta)    # when target = 1, this will become 0

        probs = torch.sigmoid(inputs)   # convert logits to probabilities

        # use logsigmoid for numerical stability
        # NOTE: log(1 - sigmoid(x)) = log(sigmoid(-x))
        pos_loss = -(1-probs)**self.alpha * F.logsigmoid(inputs) * pos_weight
        neg_loss = -probs**self.alpha * F.logsigmoid(-inputs) * neg_weight

        loss = pos_loss + neg_loss

        if self.reduction == "sum":
            return torch.sum(loss)
        
        if self.reduction == "mean":
            return torch.sum(loss) / torch.sum(pos_weight)

        return loss

class QualityFocalLossWithLogits(nn.Module):
    """Implement Quality Focal Loss (from Generalized Focal Loss v1) with Logits to improve numerical stability. Default values are from the original paper.

    Args:
        beta: control the scaling/modulating factor to reduce the impact of easy examples
        reduction: either none, sum, or mean 
    """
    def __init__(self, beta: float = 2, reduction: str = "sum"):
        super().__init__()
        assert reduction in ("none", "sum", "mean")
        self.beta = beta
        self.reduction = reduction

    def __format__(self, inputs: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(inputs)

        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        modulating_factor = torch.abs(targets - probs)**self.beta

        loss = modulating_factor * ce_loss

        if self.reduction == "sum":
            return torch.sum(loss)
        
        if self.reduction == "mean":
            return torch.sum(loss) / targets.eq(1).float().sum()

        return loss
