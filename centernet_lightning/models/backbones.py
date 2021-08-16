from typing import Dict, Union
from copy import deepcopy

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models import resnet, mobilenet

try:
    import timm
except ImportError:
    pass

from ..utils import load_config

_backbone_channels = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "resnet101": [64, 256, 512, 1024, 2048],

    "mobilenet_v2": [16, 24, 32, 96, 320],
    "mobilenet_v3_small": [16, 16, 24, 48, 96],
    "mobilenet_v3_large": [16, 24, 40, 112, 160]
}

class ResNetBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, return_features=False, frozen_stages=0, freeze_bn_stats=False, **kwargs):
        super().__init__()
        self.frozen_stages = frozen_stages
        self.freeze_bn_stats = freeze_bn_stats
        backbone = resnet.__dict__[name](pretrained=pretrained)

        # may replace 7x7 conv with 3 3x3 conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.stage1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

        self.out_channels = _backbone_channels[name]
        self.return_features = return_features
        self.output_stride = 32

        self.freeze_stages()

    def forward(self, x):
        out1 = self.stem(x)         # stride 2
        out2 = self.stage1(out1)    # stride 4
        out3 = self.stage2(out2)    # stride 8
        out4 = self.stage3(out3)    # stride 16
        out5 = self.stage4(out4)    # stride 32

        if self.return_features:
            return [out1, out2, out3, out4, out5]
        
        return out5

    # mmdetection trick
    # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L612
    def freeze_stages(self):
        stages = [self.stem, self.stage1, self.stage2, self.stage3, self.stage4]
        for i in range(self.frozen_stages):
            stages[i].eval()        # set to evaluation mode so BN won't update running statistics
            for param in stages[i].parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode=mode)
        self.freeze_stages()
        if self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()

class MobileNetBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, return_features=False, frozen_stages=0, freeze_bn_stats=False, **kwargs):
        super().__init__()
        self.frozen_stages = frozen_stages
        self.freeze_bn_stats = freeze_bn_stats

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
        
        # NOTE: the last expansion 1x1 conv is not included to save computation
        # MobileNetV2 paper does this for DeepLabV3 - MobileNetV2
        self.stages.append(nn.Sequential(*stage))

        self.out_channels = _backbone_channels[name]
        self.return_features = return_features
        self.output_stride = 32

        self.freeze_stages()

    def forward(self, x):
        out = []
        next = x
        for stage in self.stages:
            next = stage(next)
            out.append(next)

        if self.return_features:
            return out
        
        return out[-1]

    # mmdetection trick
    # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L612
    def freeze_stages(self):
        for i in range(self.frozen_stages):
            self.stages[i].eval()       # set to evaluation mode so BN won't update running statistics
            for param in self.stages[i].parameters():
                param.requires_grad = False

    def train(self, mode):
        super().train(mode=mode)
        self.freeze_stages()
        if self.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()

class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, return_features=False, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True)
        
        self.out_channels = self.backbone.feature_info.channels()
        self.return_features = return_features
        self.output_stride = self.backbone.feature_info.reduction()[-1]

    def forward(self, x):
        out = self.backbone(x)
        
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

    elif config["name"].startswith("timm"):
        config = deepcopy(config)
        config["name"] = config["name"][len("timm_"):]
        backbone = TimmBackbone(**config, return_features=return_features)

    else:
        raise "Backbone not supported"
    
    return backbone
