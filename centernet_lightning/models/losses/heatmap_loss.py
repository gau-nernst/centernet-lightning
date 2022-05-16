import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    'CornerNetFocalLoss', 'QualityFocalLoss'
]


# reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/gaussian_focal_loss.py
class CornerNetFocalLoss(nn.Module):
    '''CornerNet Focal Loss. Paper: https://arxiv.org/abs/1808.01244
    
    Loss = (1-p)^a * log(p) if y = 1
         = (1-y)^b * p^a * log(1-p) otherwise (y < 1)
    '''
    def __init__(self, alpha: float = 2., beta: float = 4., reduction: str = "mean"):
        """CornerNet Focal Loss. Default values are from the paper

        Args:
            alpha (float): control the modulating factor to reduce the impact of easy examples. This is gamma in the original Focal loss. Default: 2
            beta (float): control the additional weight for negative examples when y < 1. Default: 4
            reduction (str): either none, sum, or mean 
        """
        super().__init__()
        assert reduction in ('none', 'sum', 'mean')
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): heatmap output, before sigmoid. Shape (N, C, H, W)
            targets (torch.Tensor): target Gaussian heatmap. Shape (N, C, H, W)
        """
        probs = torch.sigmoid(inputs)               # convert logits to probabilities
        pos_weight = (targets == 1).float()         # gaussian peaks are positive samples
        neg_weight = (1 - targets) ** self.beta     # when target = 1, this will be 0

        # NOTE: log(1 - sigmoid(x)) = log(sigmoid(-x))
        pos_loss = -pos_weight * (1 - probs) ** self.alpha * F.logsigmoid(inputs)
        neg_loss = -neg_weight * probs ** self.alpha * F.logsigmoid(-inputs)

        loss = pos_loss + neg_loss
        if self.reduction == 'none':
            return loss
        
        if self.reduction == "sum":
            return loss.sum()

        # sum over CHW dim (within each image)
        # remove samples without detections
        num_pos = pos_weight.flatten(1).sum(1)
        mask = num_pos > 0
        if mask.sum() == 0:
            return torch.tensor(0, dtype=inputs.dtype, device=inputs.device)

        loss = loss[mask].flatten(1).sum(1)
        num_pos = num_pos[mask]
        return (loss / num_pos).mean()


class QualityFocalLoss(nn.Module):
    '''Quality Focal Loss. Paper: https://arxiv.org/abs/2006.04388

    Loss = (y - p)^b * BCE(p, y)
    '''
    def __init__(self, beta: float = 2., reduction: str = 'mean'):
        '''Quality Focal Loss. Default values are from the paper

        Args:
            beta: control the scaling/modulating factor to reduce the impact of easy examples
            reduction: either none, sum, or mean 
        '''
        super().__init__()
        assert reduction in ('none', 'sum', 'mean')
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(inputs)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        modulating_factor = (targets - probs).abs() ** self.beta
        
        loss = modulating_factor * bce_loss
        if self.reduction == 'none':
            return loss

        if self.reduction == "sum":
            return loss.sum()

        num_pos = (targets == 1).flatten(1).sum(1)
        mask = num_pos > 0
        if mask.sum() == 0:
            return torch.tensor(0, dtype=inputs.dtype, device=inputs.device)
        
        loss = loss[mask].flatten(1).sum(1)
        num_pos = num_pos[mask]
        return (loss / num_pos).mean()
