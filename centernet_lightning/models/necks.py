from typing import Dict, Union

import torch
from torch import nn
import torch.nn.functional as F

from .layers import Fuse, make_conv, make_upsample
from ..utils import load_config

class SimpleNeck(nn.Module):
    """(conv + upsample) a few times (first proposed in PoseNet https://arxiv.org/abs/1804.06208)

    Equations
        stride 16: out_4 = up(conv(in_5))
        stride 8: out_3 = up(conv(out_4))
        stride 4: out_2 = up(conv(out_3))
    """
    def __init__(self, backbone_channels, upsample_channels=[256, 128, 64], conv_type="normal", upsample_type="conv_transpose", **kwargs):
        super().__init__()
        layers = []

        # first (conv + upsample) from backbone
        conv_layer = make_conv(backbone_channels[-1], upsample_channels[0], conv_type=conv_type)
        up_layer = make_upsample(upsample_type, upsample_channels[0], **kwargs)
        layers.append(conv_layer)
        layers.append(up_layer)

        for i in range(1, len(upsample_channels)):
            conv_layer = make_conv(upsample_channels[i-1], upsample_channels[i], conv_type=conv_type)
            up_layer = make_upsample(upsample_type, upsample_channels[i], **kwargs)
            layers.append(conv_layer)
            layers.append(up_layer)

        self.upsample = nn.Sequential(*layers)
        self.out_channels = upsample_channels[-1]
        self.upsample_stride = 2**len(upsample_channels)

    def forward(self, x):
        out = self.upsample(x)
        return out

class FPNNeck(nn.Module):
    """FPN neck with some modifications. Paper: https://arxiv.org/abs/1612.03144
        - Weighted fusion is used in Bi-FPN: https://arxiv.org/abs/1911.09070
        - Fusion factor (same as weighted fusion): https://arxiv.org/abs/2011.02298

    Equations
        stride 32: out_5 = conv_skip(in_5)
        stride 16: out_4 = conv(skip(in_4) + up(out_5) x w_4)
        stride 8: out_3 = conv(skip(in_3) + up(out_4) x w_3)
        stride 4: out_2 = conv(skip(in_2) + up(out_3) x w_2)
    """
    def __init__(self, backbone_channels, upsample_channels=[256, 128, 64], upsample_type="nearest", conv_type="normal", weighted_fusion=False, **kwargs):
        super().__init__()
        self.top_conv = nn.Conv2d(backbone_channels[-1], upsample_channels[0], 1)
        self.skip_connections = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        if weighted_fusion:
            # indexing ParameterList of scalars might be slightly faster than indexing Parameter of 1-d tensor
            self.weights = [nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True)) for _ in range(len(upsample_channels))]
            self.weights = nn.ParameterList(self.weights)
        else:
            self.weights = None

        for i in range(len(upsample_channels)):
            # build skip connections
            in_channels = backbone_channels[-2-i]
            out_channels = upsample_channels[i]
            skip_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.skip_connections.append(skip_conv)

            # build upsample layers
            upsample = make_upsample(upsample_type=upsample_type, deconv_channels=out_channels, **kwargs)
            self.up_layers.append(upsample)

            # build output conv layers
            out_conv_channels = upsample_channels[i+1] if i < len(upsample_channels)-1 else upsample_channels[-1]
            conv = make_conv(out_channels, out_conv_channels, conv_type=conv_type, **kwargs)
            self.conv_layers.append(conv)

        self.out_channels = upsample_channels[-1]
        self.upsample_stride = 2**len(upsample_channels)

    def forward(self, features):
        out = features[-1]
        out = self.top_conv(out)
        
        for i in range(len(self.conv_layers)):
            skip = self.skip_connections[i](features[-2-i]) # skip connection
            up = self.up_layers[i](out)                     # upsample
            
            if self.weights is not None:
                w = F.relu(self.weights[i])
                out = (skip + up*w) / (1 + w)       # combine with fusion weight
            else:
                out = skip + up
            out = self.conv_layers[i](out)          # output conv

        return out

class IDANeck(nn.Module):
    """IDA neck used in Deep Layer Aggregation. Paper: https://arxiv.org/abs/1707.06484
    
        backbone: [256, 512, 1024, 2048]
        layer 1: [64, 128, 256]
        layer 2: [64, 128]
        layer 3: [64]
    """
    def __init__(self, backbone_channels, upsample_channels=[256, 128, 64], upsample_type="nearest", conv_type="normal", weighted_fusion=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n = len(upsample_channels)

        last_channels = backbone_channels[-1:-2-self.n:-1]  # reverse list
        out_channels = upsample_channels
        for _ in range(self.n):
            fuse_nodes = nn.ModuleList()
            for j in range(len(out_channels)):
                top = last_channels[j]
                lateral = last_channels[j+1]
                out = out_channels[j]

                fuse = Fuse([lateral, top], out, "up", upsample=upsample_type, conv_type=conv_type, weighted_fusion=weighted_fusion)
                fuse_nodes.append(fuse)   

            self.layers.append(fuse_nodes)
            last_channels = out_channels
            out_channels = out_channels[1:]

        self.out_channels = upsample_channels[-1]
        self.upsample_stride = 2**len(upsample_channels)

    def forward(self, features):
        out = features[-1:-2-self.n:-1]     # reverse list
        for fuse_nodes in self.layers:
            out = [fuse_nodes[i](out[i+1], out[i]) for i in range(len(fuse_nodes))]
        return out[0]

class BiFPNLayer(nn.Module):
    """"""
    def __init__(self, num_features=4, num_channels=64, upsample_type="nearest", downsample_type="max", conv_type="normal", weighted_fusion=True, **kwargs):
        super().__init__()
        assert isinstance(num_channels, int)
        self.num_features = num_features
        self.top_down = nn.ModuleList()
        self.bottom_up = nn.ModuleList()

        # build top down
        for _ in range(num_features-1):
            fuse = Fuse([num_channels]*2, num_channels, "up", upsample=upsample_type, conv_type=conv_type, weighted_fusion=weighted_fusion)
            self.top_down.append(fuse)

        # build bottom up
        for _ in range(1, num_features-1):
            fuse = Fuse([num_channels]*3, num_channels, "down", downsample=downsample_type, conv_type=conv_type, weighted_fusion=weighted_fusion)
            self.bottom_up.append(fuse)

        self.last_fuse = Fuse([num_channels]*2, num_channels, "down", downsample=downsample_type, conv_type=conv_type, weighted_fusion=weighted_fusion)

    def forward(self, features):
        # top down: Ptd_6 = conv(Pin_6 + up(Ptd_7))
        topdowns = [None] * len(features)
        topdowns[-1] = features[-1]
        for i in range(len(self.top_down)):
            topdowns[-2-i] = self.top_down[i](features[-2-i], topdowns[-1-i])
        
        # bottom up: Pout_6 = conv(Pin_6 + Ptd_6 + down(Pout_5))
        out = [None] * len(features)
        out[0] = topdowns[0]
        for i in range(len(self.bottom_up)):
            out[i+1] = self.bottom_up[i](features[i+1], topdowns[i+1], out[i])
        out[-1] = self.last_fuse(features[-1], out[-2])
        
        return out

class BiFPNNeck(nn.Module):
    def __init__(self, backbone_channels, num_layers=3, num_features=4, num_channels=64, upsample_type="nearest", downsample_type="max", conv_type="normal", weighted_fusion=True, **kwargs):
        super().__init__()
        self.project = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.num_features = num_features

        for b_channels in backbone_channels[-num_features:]:
            conv = nn.Conv2d(b_channels, num_channels, 1)
            self.project.append(conv)

        for _ in range(num_layers):
            bifpn_layer = BiFPNLayer(num_features=num_features, num_channels=num_channels, upsample_type=upsample_type, downsample_type=downsample_type, conv_type=conv_type, weighted_fusion=weighted_fusion, **kwargs)
            self.layers.append(bifpn_layer)

        self.out_channels = num_channels
        self.upsample_stride = 2**(num_features-1)

    def forward(self, features):
        out = [project(x) for project, x in zip(self.project, features[-self.num_features:])]

        for bifpn_layer in self.layers:
            out = bifpn_layer(out)
        
        return out[0]

def build_neck(config: Union[str, Dict], backbone_channels):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]["neck"]

    neck_mapper = {
        "simple": SimpleNeck,
        "fpn": FPNNeck,
        "ida": IDANeck,
        "bifpn": BiFPNNeck
    }

    if config["name"] in neck_mapper:
        neck = neck_mapper[config["name"]](backbone_channels, **config)

    else:
        raise "Neck not supported"
    
    return neck
