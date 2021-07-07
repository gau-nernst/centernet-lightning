from typing import Union, Dict
import os

from torch import nn

from backbones import ResNetBackbone, MobileNetBackbone
from necks import SimpleNeck, FPNNeck
from heads import HeatmapHead, Box2DHead
from centernet import CenterNet
from ..utils import load_config

def build_backbone(config: Union[str, Dict], return_features=False):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["backbone"]
    
    if config["name"].startswith("resnet"):
        backbone = ResNetBackbone(**config, return_features=return_features)

    elif config["name"].startswith("mobilenet"):
        backbone = MobileNetBackbone(**config["name"], return_features=return_features)

    else:
        raise "Backbone not supported"
    
    return backbone

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

def build_output_heads(config: Union[str, Dict], in_channels):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["output_heads"]

    output_heads = nn.ModuleDict()
    output_head_mapper = {
        "heatmap": HeatmapHead,
        "box2d": Box2DHead
    }
    for name, params in config.items():
        head = output_head_mapper[name](in_channels, **params)
        output_heads[name] = head
    
    return output_heads

def build_centernet(config: Union[str, Dict]):
    """Build CenterNet from config. Either path to config file or a dictionary
    """
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]

    model = CenterNet(**config)
    return model
