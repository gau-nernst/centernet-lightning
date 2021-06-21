import torch
from torch import nn
from torchvision.models import mobilenet

from .base import get_resnet_stages, get_mobilenet_stages
from .simple import SimpleBackbone
from .fpn import FPNBackbone

_resnet_channels = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
    "resnet101": [256, 512, 1024, 2048]
}

_mobilenet_channels = {
    "mobilenet_v2": [16, 24, 32, 96, 1280],
    "mobilenet_v3_small": [16, 16, 24, 48, 576],
    "mobilenet_v3_large": [16, 24, 40, 112, 960]
}

def build_simple_backbone(name: str, pretrained: bool = True, **kwargs):
    if name.startswith("resnet"):
        resnet_stages = get_resnet_stages(name, pretrained)
        downsample = nn.Sequential(*resnet_stages)
        last_channels = _resnet_channels[name][-1]

    elif name.startswith("mobilenet"):
        backbone = mobilenet.__dict__[name](pretrained=pretrained)
        downsample = backbone.features
        last_channels = _mobilenet_channels[name][-1]

    else:
        raise ValueError(f"{name} is not supported")

    return SimpleBackbone(downsample, last_channels, **kwargs)

def build_fpn_backbone(name: str, pretrained: str = True, **kwargs):
    if name.startswith("resnet"):
        resnet_stages = get_resnet_stages(name, pretrained)
        downsample = nn.ModuleList(resnet_stages)
        downsample_channels = _resnet_channels[name]
    
    elif name.startswith("mobilenet"):
        mobilenet_stages = get_mobilenet_stages(name, pretrained)
        downsample = nn.ModuleList(mobilenet_stages)
        downsample_channels = _mobilenet_channels[name]

    else:
        raise ValueError(f"{name} is not supported")

    return FPNBackbone(downsample, downsample_channels, **kwargs)

def build_backbone(name: str, pretrained: str = True, **kwargs):
    if name.endswith("-fpn"):
        name = name[:-4]    # remove "-fpn"
        return build_fpn_backbone(name, pretrained=pretrained, **kwargs)
    
    return build_simple_backbone(name, pretrained=pretrained, **kwargs)
