from typing import Iterable, Tuple
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

def render_gaussian_kernel(
    heatmap: torch.Tensor,
    center_x: float,
    center_y: float,
    box_w: float,
    box_h: float,
    alpha: float=0.54
    ):
    """Reference implementation https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L241
    """

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

def reference_focal_loss(pred, gt):
    """ Reference implementation from CenterNet-better-plus https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet.py#L56
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss