import warnings
from typing import Dict, Iterable
import math

import torch
from torch import nn

class ConvUpsampleBlock(nn.Module):
    """Upsample block (convolution transpose) with optional DCN (currently not supported)

    Architecture: conv + conv transpose, with BN and relu
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_type: str = "conv_transpose",
        conv_type: str = "normal",
        deconv_params: Dict = None,
        init_bilinear: bool = True,
        **kwargs
    ):
        super(ConvUpsampleBlock, self).__init__()
        if conv_type not in ["dcn", "separable", "normal"]:
            warnings.warn(f"{conv_type} is not supported. Fall back to normal convolution")
            conv_type = "normal"
        
        if deconv_params is None:
            deconv_params = dict(kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)

        if conv_type == "dcn":
            # potential dcn implementations
            # torchvision: https://pytorch.org/vision/stable/ops.html#torchvision.ops.deform_conv2d
            # detectron: https://detectron2.readthedocs.io/en/latest/modules/layers.html#detectron2.layers.ModulatedDeformConv
            # mmcv: https://mmcv.readthedocs.io/en/stable/api.html#mmcv.ops.DeformConv2d
            raise NotImplementedError()
        elif conv_type == "separable":
            raise NotImplementedError()
        else:
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )
        
        self.conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if upsample_type not in ["conv_transpose", "bilinear", "nearest"]:
            warnings.warn(f"{upsample_type} is not supported. Fall back to convolution transpose")
            conv_type = "conv_transpose"

        if upsample_type == "conv_transpose":
            upsample = nn.ConvTranspose2d(out_channels, out_channels, **deconv_params)
        else:
            upsample = nn.Upsample(scale_factor=2, mode=upsample_type)

        # default behavior, initialize weights to bilinear upsampling
        # TF CenterNet does not do this
        if upsample_type == "conv_transpose" and init_bilinear:
            self.init_bilinear_upsampling(upsample)

        self.upsample = nn.Sequential(
            upsample,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        return out

    def init_bilinear_upsampling(self, deconv_layer):
        # initialize convolution transpose layer as bilinear upsampling
        # https://github.com/ucbdrive/dla/blob/master/dla_up.py#L26-L33
        w = deconv_layer.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2*f - 1 - f%2) / (f*2.)
        
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0,0,i,j] = (1 - math.fabs(i/f - c)) * (1 - math.fabs(j/f - c))
        
        for c in range(1, w.size(0)):
            w[c,0,:,:] = w[0,0,:,:]

class SimpleBackbone(nn.Module):
    """ResNet/MobileNet with upsample stage (first proposed in PoseNet https://arxiv.org/abs/1804.06208)
    """
    # Reference implementations
    # CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py
    # Original: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
    # TensorFlow: https://github.com/tensorflow/models/blob/master/research/object_detection/models/center_net_re.py
    # upsample parameters (channels and kernels) are from CenterNet

    def __init__(
        self, 
        downsample: nn.Module, 
        downsample_out_channels: int, 
        upsample_channels: Iterable = [256, 128, 64],
        conv_upsample_block: Dict = None,
        **kwargs
        ):
        super(SimpleBackbone, self).__init__()
        self.downsample = downsample

        # fill default values
        if conv_upsample_block is None:
            conv_upsample_block = dict(upsample_type="conv_transpose", conv_type="normal", deconv_params=None, init_bilinear=True)

        # build upsample stage
        conv_upsample_layers = []
        conv_up_layer = ConvUpsampleBlock(
            downsample_out_channels,
            upsample_channels[0],
            **conv_upsample_block
        )
        conv_upsample_layers.append(conv_up_layer)

        for i in range(1, len(upsample_channels)):
            conv_up_layer = ConvUpsampleBlock(
                upsample_channels[i-1],
                upsample_channels[i],
                **conv_upsample_block
            )
            conv_upsample_layers.append(conv_up_layer)

        self.upsample = nn.Sequential(*conv_upsample_layers)

        self.out_channels = upsample_channels[-1]

    def forward(self, x):
        out = self.downsample(x)
        out = self.upsample(out)

        return out

    @property
    def output_stride(self):
        sample_input = torch.rand((4,3,512,512))
        sample_output = self(sample_input)
        return sample_input.shape[-1] // sample_output.shape[-1]
