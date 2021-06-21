import torch
from torch import nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    """Implement Modified Focal Loss with Logits to improve numerical stability. This is originally from CornerNet
    """
    # reference implementations
    # https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/model/losses.py#L72
    # default alpha and beta values taken from CenterTrack
    def __init__(self, alpha: float=2., beta: float=4., reduction="mean"):
        super(FocalLossWithLogits, self).__init__()
        assert reduction in ("mean", "sum", "none")
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # NOTE: targets is a 2D Gaussian
        pos_mask = targets.eq(1).float()
        neg_mask = targets.lt(1).float()

        probs = torch.sigmoid(inputs)   # convert logits to probabilities

        # use logsigmoid for numerical stability
        pos_loss = -F.logsigmoid(inputs) * (1-probs)**self.alpha * pos_mask                         # loss at Gaussian peak
        neg_loss = -F.logsigmoid(-inputs) * probs**self.alpha * (1-targets)**self.beta * neg_mask   # loss at everywhere else

        if self.reduction == "none":
            return pos_loss + neg_loss

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if self.reduction == "sum":
            return pos_loss + neg_loss

        N = pos_mask.sum()  # number of peaks = number of ground-truth detections
        # use N + eps instead of 2 cases?
        if N == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / N

        return loss
