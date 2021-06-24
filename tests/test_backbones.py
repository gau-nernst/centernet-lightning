import torch
from src.backbones import SimpleBackbone, FPNBackbone, build_simple_backbone, build_fpn_backbone, build_backbone

_backbone_names = ["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"]

sample_input = torch.rand((4,3,512,512))

class TestConvUpBlock:
    pass

class TestBuilder:

    def test_build_simple_backbone(self):
        for name in _backbone_names:
            print(f"Testing simple backbone for {name}")
            backbone = build_simple_backbone(name)
            assert isinstance(backbone, SimpleBackbone)

            output = backbone(sample_input)
            assert output.shape == (4,64,128,128)
            assert backbone.output_stride == 4
 
    def test_build_fpn_backbone(self):
        for name in _backbone_names:
            print(f"Testing fpn backbone for {name}")
            backbone = build_fpn_backbone(name)
            assert isinstance(backbone, FPNBackbone)

            output = backbone(sample_input)
            assert output.shape == (4,64,128,128)
            assert backbone.output_stride == 4

    def test_build_backbone(self):
        backbone = build_backbone("resnet18")
        assert isinstance(backbone, SimpleBackbone)

        backbone = build_backbone("resnet18-fpn")
        assert isinstance(backbone, FPNBackbone)

class TestWithConfig:
    pass
