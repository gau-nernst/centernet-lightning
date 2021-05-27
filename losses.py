import torch
from torch import nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    """Implement Focal Loss with Logits to improve numerical stability. This is CornerNet version, which is used in CenterNet and CenterTrack
    """
    # reference implementations
    # https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/model/losses.py#L72
    # https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html (only for negative samples, RetinaNet version)
    # default alpha and beta values taken from CenterTrack
    def __init__(self, alpha: float=2., beta: float=4.):
        super(FocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # NOTE: targets is a 2D Gaussian
        pos_mask = (targets == 1).float()
        neg_mask = (targets < 1).float()

        probs = F.sigmoid(inputs)   # convert logits to probabilities

        # use logsigmoid for numerical stability
        pos_loss = -F.logsigmoid(inputs) * (1-probs)**self.alpha * pos_mask  # loss at Gaussian peak
        neg_loss = -F.logsigmoid(-inputs) * probs**self.alpha * (1-targets)**self.beta * neg_mask   # loss at everywhere else

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        N = pos_mask.sum()  # number of peaks = number of ground-truth detections
        if N == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / N

        return loss

def gather_feature(feature: torch.Tensor, index: torch.Tensor, mask=None):
    # (N, C, H, W) to (N, HxW, C)
    batch_dim, channel_dim = feature.shape[:2]
    feature = feature.view(batch_dim, channel_dim, -1).permute((0,2,1)).contiguous()

    dim = feature.shape[-1]
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    feature = feature.gather(dim=1, index=index)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feature)
        feature = feature[mask]
        feature = feature.reshape(-1, dim)
    return feature

def reg_l1_loss(input, mask, index, target):
    """
    mask: detection point
    """
    raise NotImplementedError()
