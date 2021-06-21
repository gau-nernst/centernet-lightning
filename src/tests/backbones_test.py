import torch
from ..backbones import *

_backbone_names = ["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]

sample_input = torch.rand((4,3,512,512))

class TestConvUpBlock:
    pass

class TestSimpleBackbone:

    def test_build_simple_backbone(self):
        for name in _backbone_names:
            print(f"Testing simple backbone for {name}")
            backbone = build_simple_backbone(name)
            
            output = backbone(sample_input)
            assert output.shape == (4,64,128,128)
            assert backbone.output_stride == 4

class TestFPNBackbone:
    
    def test_build_fpn_backbone(self):
        for name in _backbone_names:
            print(f"Testing fpn backbone for {name}")
            backbone = build_fpn_backbone(name)
            
            output = backbone(sample_input)
            assert output.shape == (4,64,128,128)
            assert backbone.output_stride == 4

class TestWithConfig:
    pass
