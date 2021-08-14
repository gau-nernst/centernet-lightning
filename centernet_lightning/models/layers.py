import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d

class DeformableConv2dBlock(nn.Module):
    # https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/example_symbol.py
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, mask_activation=None, version=2, mask_init_bias=0):
        super().__init__()
        kernel_size = _pair(kernel_size)
        if mask_activation is None:
            mask_activation = nn.Sigmoid
        elif isinstance(mask_activation, str):
            mask_activation = nn.__dict__[mask_activation]
        
        num_locations = kernel_size[0] * kernel_size[1]
        self.offset_conv = nn.Conv2d(in_channels, 2*num_locations, kernel_size, stride=stride, padding=padding)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_locations, kernel_size, stride=stride, padding=padding),
            mask_activation()
        ) if version == 2 else None
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        if self.mask_conv is not None:
            nn.init.constant_(self.mask_conv[0].weight, 0)
            nn.init.constant_(self.mask_conv[0].bias, mask_init_bias)
        
    def forward(self, input):
        offset = self.offset_conv(input)
        mask = self.mask_conv(input) if self.mask_conv is not None else None

        out = self.deform_conv(input, offset, mask)
        return out

def make_conv(in_channels, out_channels, conv_type="normal", kernel_size=3, mask_activation=None, version=2, mask_init_bias=0, depth_multiplier=1, **kwargs):
    """Create a convolution layer. Options: deformable, separable, or normal convolution
    """
    assert conv_type in ("deformable", "separable", "normal")
    padding = (kernel_size-1)//2

    if conv_type == "deformable":
        conv_layer = nn.Sequential(
            DeformableConv2dBlock(
                in_channels, out_channels, kernel_size, padding=padding, bias=False,
                mask_activation=mask_activation, version=version, mask_init_bias=mask_init_bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    elif conv_type == "separable": 
        hidden_channels = in_channels * depth_multiplier
        conv_layer = nn.Sequential(
            # dw
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        nn.init.kaiming_normal_(conv_layer[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(conv_layer[3].weight, mode="fan_out", nonlinearity="relu")
        
    else:                           # normal convolution
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(conv_layer[0].weight, mode="fan_out", nonlinearity="relu")

    return conv_layer

def make_upsample(upsample_type="nearest", deconv_channels=None, deconv_kernel=3, deconv_init_bilinear=True, **kwargs):
    """Create an upsample layer. Options: convolution transpose, bilinear upsampling, or nearest upsampling
    """
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
    """Initialize convolution transpose layer as bilinear upsampling to help with training stability
    """ 
    # https://github.com/ucbdrive/dla/blob/master/dla_up.py#L26-L33
    w = deconv_layer.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2*f - 1 - f%2) / (f*2.)
    
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0,0,i,j] = (1 - math.fabs(i/f - c)) * (1 - math.fabs(j/f - c))
    
    for c in range(1, w.size(0)):
        w[c,0,:,:] = w[0,0,:,:]

def make_downsample(downsample_type="max", conv_channels=None, conv_kernel=3, **kwargs):
    """Create a downsample layer. Options: convolution, max pooling, or average pooling
    """
    assert downsample_type in ("max", "average", "conv")

    if downsample_type == "conv":
        downsample = nn.Conv2d(conv_channels, conv_channels, conv_kernel, stride=2, padding="same", bias=False)
        bn = nn.BatchNorm2d(conv_channels)
        relu = nn.ReLU(inplace=True)
        downsample_layer = nn.Sequential(downsample, bn, relu)

        nn.init.kaiming_normal_(downsample.weight, mode="fan_out", nonlinearity="relu")
    
    elif downsample_type == "max":
        downsample_layer = nn.MaxPool2d(2, 2)
    else:
        downsample_layer = nn.AvgPool2d(2, 2)
    
    return downsample_layer

class Fuse(nn.Module):
    """Fusion node to be used for feature fusion. To be used in `BiFPNNeck` and `IDANeck`. The last input will be resized.

    Formula
        no weight: out = conv(in1 + resize(in2))
        weighted: out = conv((in1*w1 + resize(in2)*w2) / (w1 + w2 + eps))
    """
    def __init__(self, in_channels, out, resize, upsample="nearest", downsample="max", conv_type="normal", weighted_fusion=False):
        super().__init__()
        assert resize in ("up", "down")
        self.project = nn.ModuleList()
        self.weights = nn.Parameter(torch.ones(len(in_channels)), requires_grad=True) if weighted_fusion else None

        for in_c in in_channels:
            project_conv = nn.Conv2d(in_c, out, 1) if in_c != out else None # match output channels
            self.project.append(project_conv)
        if resize == "up":
            self.resize = make_upsample(upsample_type=upsample, deconv_channels=out)
        else:
            self.resize = make_downsample(downsample=downsample, conv_channels=out)
        self.output_conv = make_conv(out, out, conv_type=conv_type)

    def forward(self, *features, eps=1e-6):
        out = []
        for project, x in zip(self.project, features):
            out.append(project(x) if project is not None else x)
        
        out[-1] = self.resize(out[-1])

        # weighted fusion
        if self.weights is not None:
            weights = F.relu(self.weights)
            out = torch.stack([out[i]*weights[i] for i in range(len(out))], dim=-1)
            out = torch.sum(out, dim=-1) / (torch.sum(weights) + eps)
        else:
            out = torch.stack(out, dim=-1)
            out = torch.sum(out, dim=-1)
        
        out = self.output_conv(out)
        return out
