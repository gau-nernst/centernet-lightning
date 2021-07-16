import torch

from src.models.necks import SimpleNeck, FPNNeck, IDANeck, BiFPNNeck, build_neck

test_config = "configs/test_config.yaml"

class TestSimpleNeck:
    def test_construction(self):
        neck = SimpleNeck([1024], upsample_channels=[256,128,64])
        assert neck.out_channels == 64
        assert neck.upsample_stride == 8

    def test_forward(self):
        neck = SimpleNeck([1024], upsample_channels=[256,128,64])
        outputs = neck(torch.rand((4,1024,16,16)))
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (4,64,128,128)

class TestFPNNeck:
    def test_construction(self):
        neck = FPNNeck([64,128,256,512,1024], upsample_channels=[256,128,64])
        assert neck.out_channels == 64
        assert neck.upsample_stride == 8

    def test_forward(self):
        neck = FPNNeck([64,128,256,512,1024], upsample_channels=[256,128,64])
        inputs = [
            torch.rand((4,128,128,128)),
            torch.rand((4,256,64,64)),
            torch.rand((4,512,32,32)),
            torch.rand((4,1024,16,16))
        ]
        outputs = neck(inputs)
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (4,64,128,128)

class TestBuilder:
    def test_build_from_file(self):
        neck = build_neck(test_config, backbone_channels=[64,128,256,512,1024])
        assert isinstance(neck, SimpleNeck)

    def test_build_from_dict(self):
        config = {"name": "simple"}
        neck = build_neck(config, backbone_channels=[1024])
        assert isinstance(neck, SimpleNeck)

        config = {"name": "fpn"}
        neck = build_neck(config, backbone_channels=[64,128,256,512,1024])
        assert isinstance(neck, FPNNeck)

        config = {"name": "ida"}
        neck = build_neck(config, backbone_channels=[64,128,256,512,1024])
        assert isinstance(neck, IDANeck)

        config = {"name": "bifpn"}
        neck = build_neck(config, backbone_channels=[64,128,256,512,1024])
        assert isinstance(neck, BiFPNNeck)
