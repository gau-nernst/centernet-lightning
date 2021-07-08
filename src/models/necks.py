import warnings
from typing import Dict, Iterable, Union
import math

import torch
from torch import nn

from ..utils import load_config

# potential dcn implementations
# torchvision: https://pytorch.org/vision/stable/ops.html#torchvision.ops.deform_conv2d
# detectron: https://detectron2.readthedocs.io/en/latest/modules/layers.html#detectron2.layers.ModulatedDeformConv
# mmcv: https://mmcv.readthedocs.io/en/stable/api.html#mmcv.ops.DeformConv2d

class ConvUpsampleBlock(nn.Module):
    """Convolution followed by Upsample
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_type: str = "conv_transpose",
        conv_type: str = "normal",
        deconv_kernel: int = 4,
        deconv_init_bilinear: bool = True,
        **kwargs
    ):
        super().__init__()
        assert conv_type in ("dcn", "separable", "normal")
        assert upsample_type in ("conv_transpose", "bilinear", "nearest")

        if conv_type == "dcn":          # deformable convolution
            raise NotImplementedError()
        elif conv_type == "separable":  # depthwise-separable convolution
            raise NotImplementedError()
        else:                           # normal convolution
            conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(conv, bn, relu)

        # convolution transpose
        if upsample_type == "conv_transpose":
            # calculate padding
            output_padding = deconv_kernel % 2
            padding = (deconv_kernel + output_padding) // 2 - 1

            upsample = nn.ConvTranspose2d(
                out_channels, out_channels, deconv_kernel, stride=2,
                padding=padding, output_padding=output_padding, bias=False
            )

            if deconv_init_bilinear:    # TF CenterNet does not do this
                self._init_bilinear_upsampling(upsample)
        
        # normal upsampling
        else:
            upsample = nn.Upsample(scale_factor=2, mode=upsample_type)

        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(upsample, bn, relu)

    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        return out

    def _init_bilinear_upsampling(self, deconv_layer):
        # initialize convolution transpose layer as bilinear upsampling
        # this helps with training stability
        # https://github.com/ucbdrive/dla/blob/master/dla_up.py#L26-L33
        w = deconv_layer.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2*f - 1 - f%2) / (f*2.)
        
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0,0,i,j] = (1 - math.fabs(i/f - c)) * (1 - math.fabs(j/f - c))
        
        for c in range(1, w.size(0)):
            w[c,0,:,:] = w[0,0,:,:]

class SimpleNeck(nn.Module):
    """ResNet/MobileNet with upsample stage (first proposed in PoseNet https://arxiv.org/abs/1804.06208)
    """
    # Reference implementations
    # CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py
    # Original: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
    # TensorFlow: https://github.com/tensorflow/models/blob/master/research/object_detection/models/center_net_re.py
    # upsample parameters (channels and kernels) are from CenterNet

    def __init__(self, backbone_channels: Iterable[int], upsample_channels: Iterable = [256, 128, 64], **kwargs):
        super().__init__()
        
        # build upsample stage
        conv_upsample_layers = []
        conv_up_layer = ConvUpsampleBlock(backbone_channels[-1], upsample_channels[0], **kwargs)
        conv_upsample_layers.append(conv_up_layer)

        for i in range(1, len(upsample_channels)):
            conv_up_layer = ConvUpsampleBlock(upsample_channels[i-1], upsample_channels[i], **kwargs)
            conv_upsample_layers.append(conv_up_layer)

        self.upsample = nn.Sequential(*conv_upsample_layers)
        self.out_channels = upsample_channels[-1]
        self.upsample_stride = 2**len(upsample_channels)

    def forward(self, x):
        out = self.upsample(x)

        return out

class FPNNeck(nn.Module):
    def __init__(self, backbone_channels: Iterable[int], upsample_channels: Iterable[int] = [256, 128, 64], skip_kernel: int = 3, **kwargs):
        """
            bottom_up_channels list from bottom to top (forward pass of the backbone)
            top_down_channels list from top to bottom (forward pass of the FPN)
        """
        super().__init__()
        self.skip_connections = nn.ModuleList()
        self.conv_upsample_layers = nn.ModuleList()

        # build skip connections
        for i in range(len(upsample_channels)):
            in_channels = backbone_channels[-2-i]
            out_channels = upsample_channels[i]

            padding = (skip_kernel - 1) // 2
            skip_conv = nn.Conv2d(in_channels, out_channels, skip_kernel, padding=padding)
            self.skip_connections.append(skip_conv)

        # build top-down layers
        conv_upsample = ConvUpsampleBlock(backbone_channels[-1], upsample_channels[0], **kwargs)
        self.conv_upsample_layers.append(conv_upsample)

        for i in range(1, len(upsample_channels)):
            conv_upsample = ConvUpsampleBlock(upsample_channels[i-1], upsample_channels[i], **kwargs)
            self.conv_upsample_layers.append(conv_upsample)

        self.out_channels = upsample_channels[-1]
        self.upsample_stride = 2**len(upsample_channels)

    def forward(self, features):
        out = features[-1]
        for i in range(len(self.conv_upsample_layers)):
            skip = self.skip_connections[i](features[-2-i])     # skip connection
            out = self.conv_upsample_layers[i](out)             # upsample
            out = out + skip                                    # combine

        return out

def build_neck(config: Union[str, Dict], backbone_channels):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["neck"]

    if config["name"] == "simple":
        neck = SimpleNeck(backbone_channels, **config)

    elif config["name"] == "fpn":
        neck = FPNNeck(backbone_channels, **config)
    
    else:
        raise "Neck not supported"
    
    return neck
