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

        probs = torch.sigmoid(inputs)   # convert logits to probabilities

        # use logsigmoid for numerical stability
        pos_loss = -F.logsigmoid(inputs) * (1-probs)**self.alpha * pos_mask                         # loss at Gaussian peak
        neg_loss = -F.logsigmoid(-inputs) * probs**self.alpha * (1-targets)**self.beta * neg_mask   # loss at everywhere else

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        N = pos_mask.sum()  # number of peaks = number of ground-truth detections
        if N == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / N

        return loss

def render_target_heatmap(
    shape: Iterable, centers: torch.Tensor, sizes: torch.Tensor, 
    indices: torch.Tensor, mask: torch.Tensor, 
    alpha: float=0.54, device: str="cpu", eps=1e-6
    ):
    """Render target heatmap using Gaussian kernel from detections' bounding boxes

    Reference implementation https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L241
    """
    heatmap = torch.zeros(shape, dtype=torch.float32, device=device)
    box_w = sizes[:,0]
    box_h = sizes[:,1]
    indices = indices.long()

    # TTFNet. add 1e-4 to variance to avoid division by zero
    std_w = alpha*box_w/6
    std_h = alpha*box_h/6
    var_w = std_w*std_w
    var_h = std_h*std_h

    # a matrix of (x,y)
    grid_y, grid_x = torch.meshgrid([
        torch.arange(shape[1], dtype=torch.float32, device=device),
        torch.arange(shape[2], dtype=torch.float32, device=device)]
    )

    # iterate over the detections
    # for i in range(len(centers)):
    for i, m in enumerate(mask):
        if m == 0:
            continue
        x = centers[i][0]
        y = centers[i][1]
        idx = indices[i]

        # gaussian kernel
        radius_sq = (x - grid_x)**2/(2*var_w[i] + eps) + (y - grid_y)**2/(2*var_h[i] + eps)
        gaussian_kernel = torch.exp(-radius_sq)
        gaussian_kernel[y, x] = 1       # force the center to be 1
        # apply mask to ignore none detections from padding
        heatmap[idx] = torch.maximum(heatmap[idx], gaussian_kernel)

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