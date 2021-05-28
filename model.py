from typing import Dict, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

import math

from losses import FocalLossWithLogits, render_gaussian_kernel

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
    """Upsample block (convolution transpose) with optional DCN

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
    supported_heads = ["size", "offset"]

    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int, 
        other_heads: Iterable[str]=None, 
        loss_weights: Iterable[float]=None,
        heatmap_bias: float=-2.7,
        batch_size: int=4, 
        lr: float=1e-3
        ):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.out_channels

        # other_heads excludes the compulsory heatmap head
        # loss weights are used to calculated total weighted loss
        # must match the number of output heads
        if other_heads == None:
            other_heads = ["size", "offset"]
        assert "heatmap" not in other_heads
        self.other_heads = other_heads

        if loss_weights == None:
            loss_weights = [0.1, 1]     # default values for centernet
        assert len(other_heads) == len(loss_weights)

        # for heatmap output, fill a pre-defined bias value
        # for other outputs, fill bias with 0 to match identity mapping (from centernet)
        self.output_heads = nn.ModuleDict()
        self.output_heads["heatmap"] = OutputHead(
            feature_channels, num_classes, 
            fill_bias=True, bias_value=heatmap_bias)
        
        for h in other_heads:
            assert h in self.supported_heads
            self.output_heads[h] = OutputHead(
                feature_channels, 2, 
                fill_bias=True, bias_value=0)
        
        self.focal_loss = FocalLossWithLogits(alpha=2., beta=4.)

        # for pytorch lightning
        self.batch_size = batch_size
        self.learning_rate = lr

    def forward(self, **kwargs):
        # print("Hello", x)
        # print(x.shape)
        features = self.backbone(kwargs["image"])
        output = {}
        for k,v in self.output_heads.items():
            # k is head name, v is that head nn.Module
            output[k] = v(features)
        
        return output

    def compute_loss(self, batch):
        img = batch["image"]
        output = self(img)
        output_h, output_w = output["heatmap"].shape[-2:]

        losses = {
            "heatmap": torch.Tensor(0., dtype=torch.float32, device=self.device)
        }
        for h in self.other_heads:
            losses[h] = torch.Tensor(0., dtype=torch.float32, device=self.device)

        target_heatmap = torch.zeros_like(output["heatmap"])

        # convert relative size to absolute size 
        batch["bboxes"][:,[0,2]] *= output_w
        batch["bboxes"][:,[1,3]] *= output_h

        # render target heatmap, size and offset maps
        for bbox, label, mask in zip(batch["bboxes"], batch["labels"], batch["mask"]):
            center_x, center_y, box_w, box_h = bbox

            # convert to integers for getting coordinates
            center_x_int = int(center_x)
            center_y_int = int(center_y)

            if "size" in output:
                losses["size"] += F.smooth_l1_loss(output["size"][0,center_y_int,center_x_int], box_w, reduction="sum")*mask
                losses["size"] += F.smooth_l1_loss(output["size"][1,center_y_int,center_x_int], box_h, reduction="sum")*mask
            if "offset" in output:
                losses["offset"] += F.smooth_l1_loss(output["offset"][0,center_y_int,center_x_int], center_x-center_x_int)*mask
                losses["offset"] += F.smooth_l1_loss(output["offset"][0,center_y_int,center_x_int], center_y-center_y_int)*mask
            if "displacement" in output:
                pass

            # render object center to target heatmap
            target_heatmap[label] = render_gaussian_kernel(target_heatmap[label], center_x_int, center_y_int, box_w, box_h)

        losses["heatmap"] = self.focal_loss(output["heatmap"], target_heatmap)

        return losses

    # lightning method, return loss here
    def training_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        
        total_loss = losses["heatmap"]
        for h in self.other_heads:
            total_loss += losses[h]

        self.log('train_loss', losses)
        return total_loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        # calculate loss and evaluation metrics

    def test_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        # same as validation

    # lightning method
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
