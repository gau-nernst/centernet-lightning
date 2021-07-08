from typing import Dict, Union

from torch import nn
from torchvision.models import resnet, mobilenet

from ..utils import load_config

_backbone_channels = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
    "resnet101": [256, 512, 1024, 2048],

    "mobilenet_v2": [16, 24, 32, 96, 1280],
    "mobilenet_v3_small": [16, 16, 24, 48, 576],
    "mobilenet_v3_large": [16, 24, 40, 112, 960]
}

class ResNetBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, return_features=False, **kwargs):
        super().__init__()
        backbone = resnet.__dict__[name](pretrained=pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        self.out_channels = _backbone_channels[name]
        self.return_features = return_features
        self.output_stride = 32

    def forward(self, x):
        out1 = self.stem(x)         # stride 4
        out2 = self.stage1(out1)    # stride 4
        out3 = self.stage2(out2)    # stride 8
        out4 = self.stage3(out3)    # stride 16
        out5 = self.stage4(out4)    # stride 32

        if self.return_features:
            return [out1, out2, out3, out4, out5]
        
        return out5

class MobileNetBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, return_features=False, **kwargs):
        super().__init__()

        # conv with stride = 2 (downsample) will be the first layer of each stage
        # this is to ensure that at each stage, it is the most refined feature map at that resolution
        # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py
        backbone = mobilenet.__dict__[name](pretrained=pretrained)
        features = backbone.features
        
        self.stages = nn.ModuleList()
        stage = [features[0]]
        
        for i in range(1, len(features)-1):
            # stride = 2, start of a new stage
            if features[i]._is_cn:
                self.stages.append(nn.Sequential(*stage))
                stage = [features[i]]
            else:
                stage.append(features[i])
        
        # include last conv layer in the last stage
        stage.append(features[-1])
        self.stages.append(nn.Sequential(*stage))

        self.out_channels = _backbone_channels[name]
        self.return_features = return_features
        self.output_stride = 32

    def forward(self, x):
        out = []
        next = x
        for stage in self.stages:
            next = stage(next)
            out.append(next)

        if self.return_features:
            return out
        
        return out[-1]

def build_backbone(config: Union[str, Dict], return_features=False):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["backbone"]
    
    if config["name"].startswith("resnet"):
        backbone = ResNetBackbone(**config, return_features=return_features)

    elif config["name"].startswith("mobilenet"):
        backbone = MobileNetBackbone(**config, return_features=return_features)

    else:
        raise "Backbone not supported"
    
    return backbone
