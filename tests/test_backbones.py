import torch
from centernet_lightning.models.backbones import ResNetBackbone, MobileNetBackbone, build_backbone

_backbones = [
    {
        "constructor": ResNetBackbone,
        "names": ["resnet18", "resnet34", "resnet50", "resnet101"],
        "output_stride": 32
    },
    {
        "constructor": MobileNetBackbone,
        "names": ["mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"],
        "output_stride": 32
    }
]

test_config = "configs/test_config.yaml"

class TestConvUpBlock:
    pass

class TestBackbones:
    def test_construction(self):
        for backbone_type in _backbones:
            constructor = backbone_type["constructor"]
            names = backbone_type["names"]

            for name in names:
                model = constructor(name)
                assert model.output_stride == backbone_type["output_stride"]
                assert isinstance(model.out_channels, list)

    def test_forward_single_output(self):
        for backbone_type in _backbones:
            constructor = backbone_type["constructor"]
            names = backbone_type["names"]

            model = constructor(names[0])
            outputs = model(torch.rand((4,3,224,224)))
            assert isinstance(outputs, torch.Tensor)

    def test_forward_multiple_outputs(self):
        for backbone_type in _backbones:
            constructor = backbone_type["constructor"]
            names = backbone_type["names"]

            model = constructor(names[0], return_features=True)
            outputs = model(torch.rand((4,3,224,224)))
            assert isinstance(outputs, list)
            for out in outputs:
                assert isinstance(out, torch.Tensor)

class TestBuilder:
    def test_build_from_file(self):
        backbone = build_backbone(test_config)
        assert isinstance(backbone, ResNetBackbone)

    def test_build_from_dict(self):
        for backbone_type in _backbones:
            constructor = backbone_type["constructor"]
            names = backbone_type["names"]
            config = {
                "name": names[0]
            }
            model = build_backbone(config)
            assert isinstance(model, constructor)
