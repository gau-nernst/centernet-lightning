import torch
from torch import nn
import torchvision.models.resnet as resnet
import torchvision.models.mobilenet as mobilenet

def get_resnet_stages(name: str, pretrained: bool = True):
    # TODO: add option to freeze stages
    backbone = resnet.__dict__[name](pretrained=pretrained)
    stages = [
        nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4
    ]
    return stages

def get_mobilenet_stages(name: str, pretrained: bool = True):
    # TODO: add option to freeze stages
    # conv with stride = 2 (downsample) will be the first layer of each stage
    # this is to ensure that at each stage, it is the most refined feature map at that re
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py
    backbone = mobilenet.__dict__[name](pretrained=pretrained)
    features = backbone.features
    stages = []

    stage = [features[0]]
    for i in range(1, len(features)-1):
        if features[i]._is_cn:      # stride = 2, start of a new stage
            stages.append(nn.Sequential(*stage))
            stage = [features[i]]
        else:
            stage.append(features[i])
    stage.append(features[-1])      # include last conv layer in the last stage
    stages.append(nn.Sequential(*stage))

    return stages
