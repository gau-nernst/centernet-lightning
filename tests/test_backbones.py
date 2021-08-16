import pytest
import torch
import yaml

from centernet_lightning.models.backbones import ResNetBackbone, MobileNetBackbone, TimmBackbone, build_backbone

@pytest.fixture
def sample_input():
    return torch.rand((4,3,512,512))

def pytest_generate_tests(metafunc):
    ids = []
    arg_names = list(metafunc.cls.backbones[0][1].keys())
    arg_values = []

    for backbone in metafunc.cls.backbones:
        backbone_id, args = backbone
        ids.append(backbone_id)
        arg_values.append(args.values())
    
    metafunc.parametrize(arg_names, arg_values, ids=ids, scope="class")

def generate_backbones():
    resnet_backbones = [(f"resnet{x}", {
        "constructor": ResNetBackbone, 
        "name": f"resnet{x}",
    }) for x in (18,34,50,101)]

    mobilenet_backbones = [(f"mobilenet_{x}", {
        "constructor": MobileNetBackbone, 
        "name": f"mobilenet_{x}",
    }) for x in ("v2", "v3_small", "v3_large")]

    timm_backbones = [(f"timm_{name}", {
        "constructor": TimmBackbone, 
        "name": name,
    }) for name in ["resnet18", "efficientnet_b0"]]

    return resnet_backbones + mobilenet_backbones + timm_backbones

class TestBackbone:
    backbones = generate_backbones()

    def test_attributes(self, constructor, name):
        model = constructor(name)
        
        assert isinstance(model.output_stride, int)
        assert isinstance(model.out_channels, list)

    def test_forward_single(self, constructor, name, sample_input):
        model = constructor(name, return_features=False)
        output = model(sample_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == model.out_channels[-1]
        assert output.shape[2] == sample_input.shape[2] // model.output_stride
        assert output.shape[3] == sample_input.shape[3] // model.output_stride

    def test_forward_multiple(self, constructor, name, sample_input):
        model = constructor(name, return_features=True)
        output = model(sample_input)
        
        assert isinstance(output, list)
        for i, (features, n_channels) in enumerate(zip(output, model.out_channels)):
            assert isinstance(features, torch.Tensor)
            assert features.shape[0] == sample_input.shape[0]
            assert features.shape[1] == n_channels
            assert features.shape[2] == sample_input.shape[2] // 2**(i+1)
            assert features.shape[3] == sample_input.shape[3] // 2**(i+1)

    def test_builder(self, constructor, name, tmp_path):
        if constructor == TimmBackbone:
            name = f"timm_{name}"

        backbone_config = {"name": name, "pretrained": True}
        backbone = build_backbone(backbone_config)
        assert isinstance(backbone, constructor)

        backbone_yaml = {"model": {"backbone": backbone_config}}
        config_path = tmp_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.dump(backbone_yaml, f)
        backbone = build_backbone(str(config_path))
        assert isinstance(backbone, constructor)
    