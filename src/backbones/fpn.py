from typing import Dict, Iterable

import torch
from torch import nn

from .simple import ConvUpsampleBlock

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
        if skip_connection is None:
            skip_connection = dict(kernel_size=1)
        
        if conv_upsample_block is None:
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

    @property
    def output_stride(self):
        sample_input = torch.rand((4,3,512,512))
        sample_output = self(sample_input)
        return sample_input.shape[-1] // sample_output.shape[-1]
