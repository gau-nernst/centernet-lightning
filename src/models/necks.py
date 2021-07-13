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

def _make_upsample(upsample_type="nearest", deconv_channels=None, deconv_kernel=3, deconv_init_bilinear=True, **kwargs):
    assert upsample_type in ("conv_transpose", "bilinear", "nearest")

    if upsample_type == "conv_transpose":
        output_padding = deconv_kernel % 2
        padding = (deconv_kernel + output_padding) // 2 - 1

        upsample = nn.ConvTranspose2d(deconv_channels, deconv_channels, deconv_kernel, stride=2, padding=padding, output_padding=output_padding, bias=False)
        bn = nn.BatchNorm2d(deconv_channels)
        relu = nn.ReLU(inplace=True)
        upsample_layer = nn.Sequential(upsample, bn, relu)

        if deconv_init_bilinear:    # TF CenterNet does not do this
            _init_bilinear_upsampling(upsample)
    
    else:
        upsample_layer = nn.Upsample(scale_factor=2, mode=upsample_type)

    return upsample_layer

def _init_bilinear_upsampling(deconv_layer):
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

def _make_conv(conv_type="normal", in_channels=None, out_channels=None, kernel_size=3, **kwargs):
    assert conv_type in ("dcn", "separable", "normal")

    if conv_type == "dcn":          # deformable convolution
        raise NotImplementedError()
    elif conv_type == "separable":  # depthwise-separable convolution
        raise NotImplementedError()
    else:                           # normal convolution
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        conv_layer = nn.Sequential(conv, bn, relu)

        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

    return conv_layer

class ConvUpsampleBlock(nn.Module):
    """Convolution followed by Upsample
    """
    def __init__(self, in_channels, out_channels, upsample_type="conv_transpose", conv_type="normal", deconv_kernel=4, deconv_init_bilinear=True, **kwargs):
        super().__init__()
        self.conv = _make_conv(conv_type, in_channels, out_channels)
        self.upsample = _make_upsample(upsample_type, deconv_channels=out_channels, deconv_kernel=deconv_kernel, deconv_init_bilinear=deconv_init_bilinear)

    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        return out

class SimpleNeck(nn.Module):
    """ResNet/MobileNet with upsample stage (first proposed in PoseNet https://arxiv.org/abs/1804.06208)
    """
    # Reference implementations
    # CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py
    # Original: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
    # TensorFlow: https://github.com/tensorflow/models/blob/master/research/object_detection/models/center_net_re.py
    # upsample parameters (channels and kernels) are from CenterNet

    def __init__(self, backbone_channels, upsample_channels=[256, 128, 64], conv_type="normal", upsample_type="conv_transpose", **kwargs):
        super().__init__()
        
        # build upsample stage
        conv_upsample_layers = []
        conv_up_layer = ConvUpsampleBlock(backbone_channels[-1], upsample_channels[0], **kwargs)
        # conv_up_layer = nn.Sequential(
        #     _make_conv(conv_type, backbone_channels[-1], upsample_channels[0]),
        #     _make_upsample(upsample_type, upsample_channels[0], **kwargs)
        # )
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
    """FPN neck with some modifications. Paper: https://arxiv.org/abs/1612.03144
        - Fusion weight: https://arxiv.org/abs/2011.02298

    Formulation
          16x16: out_5 = conv_skip(in_5)
          32x32: out_4 = conv(skip(in_4) + up(out_5) x w_4)
          64x64: out_3 = conv(skip(in_3) + up(out_4) x w_3)
        128x128: out_2 = conv(skip(in_2) + up(out_3) x w_2)
    """
    def __init__(self, backbone_channels, upsample_channels=[256, 128, 64], upsample_type="nearest", conv_type="normal", use_fusion_weights=False, **kwargs):
        super().__init__()
        self.top_conv = nn.Conv2d(backbone_channels[-1], upsample_channels[0], 1)
        self.skip_connections = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.fusion_weights = nn.ParameterList()

        for i in range(len(upsample_channels)):
            # build skip connections
            in_channels = backbone_channels[-2-i]
            out_channels = upsample_channels[i]
            skip_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.skip_connections.append(skip_conv)

            # build upsample layers
            upsample = _make_upsample(upsample_type=upsample_type, deconv_channels=out_channels, **kwargs)
            self.up_layers.append(upsample)

            # build output conv layers
            out_conv_channels = upsample_channels[i+1] if i < len(upsample_channels)-1 else upsample_channels[-1]
            conv = _make_conv(conv_type=conv_type, in_channels=out_channels, out_channels=out_conv_channels, **kwargs)
            self.conv_layers.append(conv)

            # build fusion weight
            fusion_w = nn.Parameter(torch.tensor(1.))
            fusion_w.requires_grad = use_fusion_weights
            self.fusion_weights.append(fusion_w)

        self.out_channels = upsample_channels[-1]
        self.upsample_stride = 2**len(upsample_channels)

    def forward(self, features):
        out = features[-1]
        out = self.top_conv(out)
        
        for i in range(len(self.conv_layers)):
            skip = self.skip_connections[i](features[-2-i]) # skip connection
            up = self.up_layers[i](out)                     # upsample
            out = skip + up * self.fusion_weights[i]        # combine with fusion weight
            out = self.conv_layers[i](out)                  # output conv

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
