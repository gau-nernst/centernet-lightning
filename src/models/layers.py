import math

import torch
from torch import nn
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    pass

def make_conv(in_channels, out_channels, conv_type="normal", kernel_size=3, **kwargs):
    """Create a convolution layer. Options: deformable, separable, or normal convolution
    """
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
