from typing import Iterable, Tuple
import torch
from torch import nn
import torch.nn.functional as F

class FocalLossWithLogits(nn.Module):
    """Implement Modified Focal Loss with Logits to improve numerical stability. This is originally from CornerNet
    """
    # reference implementations
    # https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/model/losses.py#L72
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

def render_target_heatmap_ttfnet(
    shape: Iterable,
    # centers: torch.Tensor,
    # sizes: torch.Tensor,
    bboxes: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor, 
    alpha: float = 0.54,
    device: str = "cpu",
    eps: float = 1e-8
    ):
    """Render target heatmap using Gaussian kernel from detections' bounding boxes. Using TTFNet method

    Reference implementation https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L241
    """
    heatmap = torch.zeros(shape, dtype=torch.float32, device=device)
    box_x = bboxes[...,0].long()
    box_y = bboxes[...,1].long()
    box_w = bboxes[...,2]
    box_h = bboxes[...,3]
    indices = indices.long()

    # From TTFNet
    var_w = torch.square(alpha * box_w / 6)
    var_h = torch.square(alpha * box_h / 6)

    # a matrix of (x,y)
    grid_y, grid_x = torch.meshgrid([
        torch.arange(shape[1], dtype=torch.float32, device=device),
        torch.arange(shape[2], dtype=torch.float32, device=device)
    ])

    # iterate over the detections
    for i, m in enumerate(mask):
        if m == 0:
            continue
        idx = indices[i]

        # gaussian kernel
        radius_sq = (box_x[i] - grid_x)**2 / (2*var_w[i] + eps) + (box_y[i] - grid_y)**2 / (2*var_h[i] + eps)
        gaussian_kernel = torch.exp(-radius_sq)
        heatmap[idx] = torch.maximum(heatmap[idx], gaussian_kernel)

    return heatmap

def render_target_heatmap_cornernet(
    shape: Iterable,
    bboxes: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    min_overlap: float = 0.7,
    device: str = "cpu",
    eps: float = 1e-8
    ):
    """Render target heatmap using Gaussian kernel from detections' bounding boxes. Using CornetNet method
    """
    # Reference implementations
    # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_gt.py
    # https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    heatmap = torch.zeros(shape, dtype=torch.float32, device=device)
    box_x = bboxes[...,0].long()
    box_y = bboxes[...,1].long()
    box_w = bboxes[...,2]
    box_h = bboxes[...,3]

    # calculate gaussian radii for all detections in an image
    radius = cornernet_gaussian_radius(box_w, box_h, min_overlap=min_overlap)
    radius = torch.clamp_min(radius, 0).long()

    diameter = 2 * radius + 1
    var = torch.square(diameter / 6)                    # sigma = diameter / 6

    for i, m in enumerate(mask):
        if m == 0:
            continue
        idx = indices[i]
        x = box_x[i]
        y = box_y[i]
        w = box_w[i]
        h = box_h[i]
        r = radius[i]

        # replace np.ogrid with torch.meshgrid since pytorch does not have ogrid
        grid_y, grid_x = torch.meshgrid([
            torch.arange(-r, r+1, dtype=torch.float32, device=device),
            torch.arange(-r, r+1, dtype=torch.float32, device=device)
        ])

        gaussian = torch.exp(-(grid_x**2 + grid_y**2) / (2*var[i] + eps))
        gaussian[gaussian < torch.finfo(gaussian.dtype).eps * torch.max(gaussian)] = 0      # clamping? is this necessary?

        left   = min(x, r)
        right  = min(w - x, r + 1).long()
        top    = min(y, r)
        bottom = min(h - y, r + 1).long()

        masked_heatmap = heatmap[idx, y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[r - top:r + bottom, r - left:r + right]
        torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap

def cornernet_gaussian_radius(width: torch.Tensor, height: torch.Tensor, min_overlap: float = 0.7):
    """Get radius for the Gaussian kernel. First used in CornerNet

    This is the bug-fixed version from CornerNet. Note that CenterNet used the bugged version
    https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    """
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return torch.min(r1, torch.min(r2, r3))

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