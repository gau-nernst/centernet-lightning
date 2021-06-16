from typing import Dict, Iterable
import warnings
import math

from torch import nn
import torchvision.models.resnet as resnet
import torchvision.models.mobilenet as mobilenet

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

def get_resnet_stages(name: str, pretrained: bool = True):
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
        
        if deconv_params == None:
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
        if conv_upsample_block == None:
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

class FPNBackbone(nn.Module):
    """
    """
    def __init__(
        self, 
        downsample: nn.ModuleList,
        downsample_channels: Iterable[int],
        upsample_channels: Iterable[int] = [256, 128, 64],
        conv_upsample_block: Dict = None,
        skip_connection: Dict = None,
        **kwargs
        ):
        """
            bottom_up_channels list from bottom to top (forward pass of the backbone)
            top_down_channels list from top to bottom (forward pass of the FPN)
        """
        super(FPNBackbone, self).__init__()
        self.downsample = downsample
        
        self.skip_connections = nn.ModuleList()
        self.conv_upsample_layers = nn.ModuleList()

        # fill default values
        if skip_connection == None:
            skip_connection = dict(kernel_size=1)
        
        if conv_upsample_block == None:
            conv_upsample_block = dict(upsample_type="conv_transpose", conv_type="normal", deconv_params=None, init_bilinear=True)

        # build skip connections
        for i in range(len(upsample_channels)):
            in_channels = downsample_channels[-2-i]
            out_channels = upsample_channels[i]

            skip_conv = nn.Conv2d(in_channels, out_channels, **skip_connection)
            self.skip_connections.append(skip_conv)

        # build first top-down layer
        conv_upsample = ConvUpsampleBlock(
            downsample_channels[-1],
            upsample_channels[0],
            **conv_upsample_block
        )
        self.conv_upsample_layers.append(conv_upsample)

        # build other top-down layers
        for i in range(1, len(upsample_channels)):
            conv_upsample = ConvUpsampleBlock(
                upsample_channels[i-1],
                upsample_channels[i], 
                **conv_upsample_block
            )
            self.conv_upsample_layers.append(conv_upsample)

        self.out_channels = upsample_channels[-1]

    def forward(self, x):
        # downsample stage. save feature maps in a list for lateral connections later
        downsample_features = [self.downsample[0](x)]
        for i in range(1, len(self.downsample)):
            next_feature = self.downsample[i](downsample_features[-1])
            downsample_features.append(next_feature)

        # upsample stage with skip connections
        out = downsample_features[-1]
        for i in range(len(self.conv_upsample_layers)):
            skip = self.skip_connections[i](downsample_features[-2-i])
            out = self.conv_upsample_layers[i](out) + skip

        return out

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