from typing import Dict, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

import math

from losses import FocalLossWithLogits, reg_l1_loss

# TODO: implement loss functions
# focal loss: https://pytorch.org/vision/stable/ops.html#torchvision.ops.sigmoid_focal_loss
# TODO: implement output decoder
# TODO: render ground-truth outputs from images
# TODO: use DLA backbone from timm

_resnet_mapper = {
    "resnet18": torchvision.models.resnet.resnet18,
    "resnet34": torchvision.models.resnet.resnet34,
    "resnet50": torchvision.models.resnet.resnet50, 
    "resnet101": torchvision.models.resnet.resnet101
}

class ResNetFeatureExtractor(nn.Module):
    """ResNet from torchvision, without output head
    """
    def __init__(self, model: str="resnet50", pretrained: bool=True):
        assert model in _resnet_mapper

        super(ResNetFeatureExtractor, self).__init__()
        backbone = _resnet_mapper[model](pretrained=pretrained)
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        return out

class UpsampleBlock(nn.Module):
    """Upsample block (convolution transpose) with optional DCN
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
        out = self.bn_conv(x)
        out = self.relu(x)

        out = self.deconv(x)
        out = self.bn_deconv(x)
        out = self.relu(x)

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
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.backbone = ResNetFeatureExtractor()
        
        resnet_output = 64
        # these are from CenterNet paper
        up_channels = [256, 128, 64]    # original PoseNet uses [256,256,256]
        up_kernels = [4, 4, 4]
        self.upsample = self._make_upsample_stage(
            in_channels=resnet_output, 
            up_channels=up_channels, 
            up_kernels=up_kernels)
        self.out_channels = up_channels[-1]

    def forward(self, x):
        features = self.backbone(x)
        features = self.upsample(features)

        return features

    def _make_upsample_stage(
        self,
        in_channels: int, 
        up_channels: List(int),
        up_kernels: List(int)) -> nn.Sequential:
        
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
    """ Output head for CenterNet. Follow this implementation https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py
    """
    def __init__(
        self, in_channels: int, out_channels: int, 
        bias_value: float=0
        ):

        super(OutputHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

        self.conv2.bias.data.fill_(bias_value)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out


class CenterNet(pl.LightningModule):
    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int, 
        heads: Iterable(str)=None, 
        loss_weights: Iterable(float)=None,
        heatmap_bias: float=-2.7,
        batch_size: int=4, 
        lr: float=1e-3
        ):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.out_channels

        # loss weights are used to calculated total weighted loss
        # must match the number of output heads
        if heads == None:
            heads = ["size", "offset"]
        if loss_weights == None:
            loss_weights = [0.1, 1]     # default values for centernet
        assert len(heads) == len(loss_weights)

        # for heatmap output, fill a pre-defined bias value
        # for other outputs, fill bias with 0 (default)
        self.output_heads = nn.ModuleDict()
        self.output_heads["heatmap"] = OutputHead(feature_channels, num_classes, bias_value=heatmap_bias)
        for h in heads:
            self.heads[h] = OutputHead(feature_channels, 2)
        
        self.heads = heads
        self.focal_loss = FocalLossWithLogits(alpha=2., beta=4.)

        # for pytorch lightning
        self.batch_size = batch_size
        self.learning_rate = lr

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        output = {}
        for k,v in self.output_heads:
            output[k] = v[features]
        
        return output

    # lightning module, define loss here
    def training_step(self, batch: Dict(torch.Tensor), batch_idx):
        x, y = batch
        output = self(x)

        losses = {}
        losses["heatmap"] = self.focal_loss(output["heatmap"], y["heatmap"])
        
        mask = None
        index = None
        for h in self.heads:
            losses[h] = reg_l1_loss(output[h], mask, index, y[h])
        
        self.log('train_loss', losses)
        return losses

    # get optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
