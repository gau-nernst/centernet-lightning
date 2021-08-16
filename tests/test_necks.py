import pytest
import torch

from centernet_lightning.models.necks import SimpleNeck, FPNNeck, IDANeck, BiFPNNeck, build_neck

@pytest.fixture
def backbone_channels():
    return [64, 64, 128, 256, 512]

@pytest.fixture
def upsample_channels():
    return [256,128,64]

@pytest.fixture
def sample_input_single():
    return torch.rand((4,512,16,16))

@pytest.fixture
def sample_input_multiple(backbone_channels):
    dims = [512 // 2**(i+1) for i in range(len(backbone_channels))]
    return [torch.rand((4,c,d,d)) for c,d in zip(backbone_channels, dims)]

class TestSimpleNeck:
    def test_attributes(self, backbone_channels, upsample_channels):
        neck = SimpleNeck(backbone_channels, upsample_channels)
       
        assert neck.out_channels == upsample_channels[-1]
        assert neck.upsample_stride == 2**len(upsample_channels)

    def test_forward(self, backbone_channels, upsample_channels, sample_input_single):
        neck = SimpleNeck(backbone_channels, upsample_channels)
        output = neck(sample_input_single)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input_single.shape[0]
        assert output.shape[1] == neck.out_channels
        assert output.shape[2] == sample_input_single.shape[2] * neck.upsample_stride
        assert output.shape[3] == sample_input_single.shape[3] * neck.upsample_stride

class TestFPNNeck:
    def test_attributes(self, backbone_channels, upsample_channels):
        neck = FPNNeck(backbone_channels, upsample_channels)

        assert neck.out_channels == upsample_channels[-1]
        assert neck.upsample_stride == 2**len(upsample_channels)

    def test_forward(self, backbone_channels, upsample_channels, sample_input_multiple):
        last_input = sample_input_multiple[-1]
        neck = FPNNeck(backbone_channels, upsample_channels)
        output = neck(sample_input_multiple)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == last_input.shape[0]
        assert output.shape[1] == neck.out_channels
        assert output.shape[2] == last_input.shape[2] * neck.upsample_stride
        assert output.shape[3] == last_input.shape[3] * neck.upsample_stride

class TestBiFPNNeck:
    pass

class TestIDANeck:
    pass
