from typing import Dict, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

import math

from losses import FocalLossWithLogits, render_target_heatmap

# TODO: implement loss functions
# TODO: implement output decoder
# TODO: render ground-truth outputs from images
# TODO: use DLA backbone from timm (is there upsample stage from timm?)

_resnet_mapper = {
    "resnet18": torchvision.models.resnet.resnet18,
    "resnet34": torchvision.models.resnet.resnet34,
    "resnet50": torchvision.models.resnet.resnet50, 
    "resnet101": torchvision.models.resnet.resnet101
}

class UpsampleBlock(nn.Module):
    """Upsample block (convolution transpose) with optional DCN (currently not supported)

    Architecture: conv + conv transpose, with BN and relu
    """
    # architecture choices
    # conv + deconv (centernet)
    # deconv + conv
    # upsampling + conv
    def __init__(
        self, in_channels: int, out_channels: int, 
        deconv_kernel: int, deconv_stride: int=2,
        deconv_pad: int=1, deconv_out_pad: int=0, 
        dcn: bool=False, init_bilinear: bool=True):
        
        super(UpsampleBlock, self).__init__()
        if dcn:
            # potential dcn implementations
            # torchvision: https://pytorch.org/vision/stable/ops.html#torchvision.ops.deform_conv2d
            # detectron: https://detectron2.readthedocs.io/en/latest/modules/layers.html#detectron2.layers.ModulatedDeformConv
            # mmcv: https://mmcv.readthedocs.io/en/stable/api.html#mmcv.ops.DeformConv2d
            raise NotImplementedError()
        
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn_conv = nn.BatchNorm2d(out_channels)

        # TODO: padding?
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, deconv_kernel, stride=deconv_stride,
            padding=deconv_pad, output_padding=deconv_out_pad, bias=False)
        self.bn_deconv = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # default behavior, initialize weights to bilinear upsampling
        if init_bilinear:
            self.init_bilinear_upsampling()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_conv(out)
        out = self.relu(out)

        out = self.deconv(out)
        out = self.bn_deconv(out)
        out = self.relu(out)

        return out

    def init_bilinear_upsampling(self):
        # initialize convolution transpose layer as bilinear upsampling
        # https://github.com/ucbdrive/dla/blob/master/dla_up.py#L26-L33
        w = self.deconv.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2*f - 1 - f%2) / (f*2.)
        
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0,0,i,j] = (1 - math.fabs(i/f - c)) * (1 - math.fabs(j/f - c))
        
        for c in range(1, w.size(0)):
            w[c,0,:,:] = w[0,0,:,:]

class ResNetBackbone(nn.Module):
    """Modified PoseNet from CenterNet https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py

    Original PoseNet https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
    """
    def __init__(self, model: str="resnet50", pretrained: bool=True):
        super(ResNetBackbone, self).__init__()
        # downsampling path from resnet
        backbone = _resnet_mapper[model](pretrained=pretrained)
        self.downsample = nn.Sequential(
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        resnet_out_channels = 2048
        
        # upsampling parameters
        # NOTE: should these be included in constructor arguments?
        up_channels = [256, 128, 64]    # from CenterNet paper. original PoseNet uses [256,256,256]
        up_kernels = [4, 4, 4]          # from PoseNet paper
        self.upsample = self._make_upsample_stage(
            in_channels=resnet_out_channels, 
            up_channels=up_channels, 
            up_kernels=up_kernels)
        self.out_channels = up_channels[-1]

    def forward(self, x):
        out = self.downsample(x)
        out = self.upsample(out)

        return out

    def _make_upsample_stage(
        self,
        in_channels: int, 
        up_channels: List[int],
        up_kernels: List[int]
        ):
        layers = []
        layers.append(UpsampleBlock(
            in_channels, up_channels[0], deconv_kernel=up_kernels[0]
        ))

        for i in range(len(up_channels)-1):
            layers.append(UpsampleBlock(
                up_channels[i], up_channels[i+1], deconv_kernel=up_kernels[i+1]
            ))

        return nn.Sequential(*layers)

class OutputHead(nn.Module):
    """ Output head for CenterNet. Reference implementation https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py
    """
    def __init__(
        self, in_channels: int, out_channels: int, 
        fill_bias: bool=False, bias_value: float=0
        ):
        super(OutputHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

        if fill_bias:
            self.conv2.bias.data.fill_(bias_value)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out

class CenterNet(pl.LightningModule):
    """General CenterNet model. Build CenterNet from a given backbone and output
    """
    supported_heads = ["size", "offset"]
    output_head_channels ={
        "size": 2,
        "offset": 2
    }

    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int, 
        other_heads: Iterable[str]=["size", "offset"], 
        loss_weights: Dict[str,float]=dict(size=0.1,offset=1),
        heatmap_bias: float=-2.7,
        max_pool_kernel: int=3,
        num_detections: int=40,
        batch_size: int=4, 
        lr: float=1e-3
        ):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.out_channels

        # for heatmap output, fill a pre-defined bias value
        # for other outputs, fill bias with 0 to match identity mapping (from centernet)
        self.output_heads = nn.ModuleDict()
        self.output_heads["heatmap"] = OutputHead(
            feature_channels, num_classes, 
            fill_bias=True, bias_value=heatmap_bias)
        # other_heads excludes the compulsory heatmap head
        for h in other_heads:
            assert h in self.supported_heads
            self.output_heads[h] = OutputHead(
                feature_channels, self.output_head_channels[h], 
                fill_bias=True, bias_value=0)
        self.other_heads = other_heads

        # loss weights are used to calculated total weighted loss
        for x in other_heads:
            assert x in loss_weights
        self.loss_weights = loss_weights   
        
        # parameterized focal loss for heatmap
        self.focal_loss = FocalLossWithLogits(alpha=2., beta=4.)

        # for detection decoding
        # this is used to mimic nms
        self.nms_max_pool = nn.MaxPool2d(max_pool_kernel, stride=1, padding=(max_pool_kernel-1)//2)     # same padding
        self.num_detections = num_detections

        # for pytorch lightning tuner
        self.batch_size = batch_size
        self.learning_rate = lr

    def forward(self, batch):
        """Return a dictionary of feature maps for each output head. Use this output to either decode to predictions or compute loss.
        """
        img = batch["image"]

        features = self.backbone(img)
        output = {}
        for k,v in self.output_heads.items():
            # k is head name, v is that head nn.Module
            output[k] = v(features)
        
        return output

    def compute_loss(self, output_maps: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Return a dictionary of losses for each output head. This method is called during the training step
        """
        bboxes = targets["bboxes"]               # ND4
        labels = targets["labels"]               # ND
        mask = targets["mask"].unsqueeze(-1)     # add column dimension to support broadcasting

        heatmap = output_maps["heatmap"]
        batch_size, channels, output_h, output_w = heatmap.shape

        size_map = output_maps["size"].view(batch_size, 2, -1)      # flatten last xy dimensions
        offset_map = output_maps["offset"].view(batch_size, 2, -1)  # for torch.gather() later

        # initialize losses to 0
        losses = {
            "heatmap": torch.tensor(0., dtype=torch.float32, device=self.device)
        }

        # convert relative size to absolute size
        # NOTE: this modifies the data in place. DO NOT run compute_loss() twice on the same data
        bboxes[:,:,[0,2]] *= output_w   # x and w
        bboxes[:,:,[1,3]] *= output_h   # y and h

        centers = bboxes[:,:,:2]        # x and y
        true_wh = bboxes[:,:,2:]        # w and h

        # convert to long so can use it as index
        # combine xy indices for torch.gather()
        centers_int = centers.long()
        center_x = centers_int[:,:,0]
        center_y = centers_int[:,:,1]
        # repeat indices using .expand() to gather on 2 channels
        xy_indices = (center_y*output_w+center_x).unsqueeze(1).expand((batch_size,2,-1))

        pred_sizes = torch.gather(size_map, dim=-1, index=xy_indices)       # N2D
        pred_offset = torch.gather(offset_map, dim=-1, index=xy_indices)    # N2D

        # need to swapaxes since pred_size is N2D but true_wh is ND2
        # use the mask to ignore none detections due to padding
        # NOTE: l1 loss can also be used here
        size_loss = F.smooth_l1_loss(pred_sizes.swapaxes(1,2), true_wh, reduction="none")
        size_loss = torch.sum(size_loss * mask)
        losses["size"] = size_loss

        offset_loss = F.smooth_l1_loss(pred_offset.swapaxes(1,2), centers - torch.floor(centers), reduction="none")
        offset_loss = torch.sum(offset_loss * mask)
        losses["offset"] = offset_loss

        for b in range(batch_size):
            # render target heatmap and accumulate focal loss
            target_heatmap = render_target_heatmap(
                heatmap.shape[1:], centers_int[b], true_wh[b], 
                labels[b], mask[b], device=self.device)
            losses["heatmap"] += self.focal_loss(heatmap[b], target_heatmap)

        # average over number of detections
        N = torch.sum(mask)
        losses["heatmap"] /= N
        losses["size"] /= N
        losses["offset"] /= N

        return losses

    def decode_detections(self, encoded_output: Dict[str, torch.Tensor]):
        """Decode model output to detections
        """
        # reference implementations
        # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L234
        # https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L118
        # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_decode.py#L28
        batch_size, channels, height, width = encoded_output["heatmap"].shape
        heatmap = encoded_output["heatmap"]
        size_map = encoded_output["size"].view(batch_size, 2, -1)        # NCHW to NC(HW)
        offset_map = encoded_output["offset"].view(batch_size, 2, -1)

        # obtain topk from heatmap
        heatmap = torch.sigmoid(encoded_output["heatmap"])  # convert to probability
        nms_mask = (heatmap == self.nms_max_pool(heatmap))  # pseudo-nms, only consider local peaks
        heatmap = nms_mask.float() * heatmap

        # flatten to N(CHW) to apply topk
        heatmap = heatmap.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(heatmap, self.num_detections)

        # restore flattened indices to class, xy indices
        topk_c_indices = topk_indices // (height*width)
        topk_xy_indices = topk_indices % (height*width)
        topk_y_indices = topk_xy_indices // width
        topk_x_indices = topk_xy_indices % width

        # extract bboxes at topk xy positions and convert to relative scales
        topk_w = torch.gather(size_map[:,0], dim=-1, index=topk_xy_indices) / width
        topk_h = torch.gather(size_map[:,1], dim=-1, index=topk_xy_indices) / height
        topk_x_offset = torch.gather(offset_map[:,0], dim=-1, index=topk_xy_indices)
        topk_y_offset = torch.gather(offset_map[:,1], dim=-1, index=topk_xy_indices)

        topk_x = (topk_x_indices + topk_x_offset) / width
        topk_y = (topk_y_indices + topk_y_offset) / height

        bboxes = torch.stack([topk_x, topk_y, topk_w, topk_h], dim=-1)  # NK4
        out = {
            "labels": topk_c_indices,
            "bboxes": bboxes,
            "scores": topk_scores
        }
        return out

    # lightning method, return total loss here
    def training_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)
        
        total_loss = losses["heatmap"]
        for h in self.other_heads:
            total_loss += losses[h] * self.loss_weights[h]

        # self.log_dict({"train": losses})
        for k,v in losses.items():
            self.log(f"train_{k}_loss", v)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)

        total_loss = losses["heatmap"]
        for h in self.other_heads:
            total_loss += losses[h] * self.loss_weights[h]

        # self.log_dict({"val": losses})
        for k,v in losses.items():
            self.log(f"val_{k}_loss", v)
        self.log("val_total_loss", total_loss)

        pred_detections = self.decode_detections(encoded_output)
        # eval_metrics = self.evaluate(pred_detections, batch)

        # calculate loss and evaluation metrics
        # log image(img, self.trainer.log_dir)
        # tensorboard = self.logger.experiment
        # tensorboard.add_image()

    def test_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)

        # self.log_dict(losses)
        for k,v in losses.items():
            self.log(f"test_{k}_loss", v)
        # same as validation

    # lightning method
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
