import pytest
import torch

from centernet_lightning.models.centernet import CenterNet, build_centernet

NUM_CLASSES = 20

@pytest.fixture
def sample_batch():
    batch_size = 4
    return {
        "image": torch.rand((batch_size,3,512,512)),
        "bboxes": torch.rand((batch_size,10,4)),
        "labels": torch.randint(0,NUM_CLASSES,(batch_size,10)),
        "mask": torch.randint(0,2,(batch_size,10))
    }

@pytest.fixture
def sample_model_outputs():
    return {
        "heatmap": torch.rand((4,80,128,128)),
        "box_2d": torch.rand((4,4,128,128)) * 128,
    }

def pytest_generate_tests(metafunc):
    ids = []
    arg_names = list(metafunc.cls.model_configs[0][1].keys())
    arg_values = []

    for config in metafunc.cls.model_configs:
        config_id, args = config
        ids.append(config_id)
        arg_values.append(args.values())
    
    metafunc.parametrize(arg_names, arg_values, ids=ids, scope="class")

def generate_model_configs():
    backbones = ["resnet18", "resnet50", "mobilenet_v2"]
    necks = ["simple", "fpn", "bifpn", "ida"]

    configs = []
    for b in backbones:
        for n in necks:
            config_id = f"{b}-{n}"
            model_config = {
                "task": "detection",
                "backbone": {"name": b},
                "neck": {"name": n},
                "output_heads": {
                    "heatmap": {"num_classes": NUM_CLASSES},
                    "box_2d": {}
                }
            }
            configs.append((config_id, model_config))

    return configs

class TestModel:
    model_configs = generate_model_configs()

    def test_attributes(self, backbone, neck, output_heads, task):
        model = CenterNet(backbone, neck, output_heads, task)

        assert isinstance(model.output_stride, int)
        assert model.task == task
        assert model.num_classes == NUM_CLASSES

    def test_get_encoded_outputs(self, backbone, neck, output_heads, task, sample_batch):
        img = sample_batch["image"]
        model = CenterNet(backbone, neck, output_heads, task)
        outputs = model.get_encoded_outputs(img)
        
        heatmap = outputs["heatmap"]
        box_2d = outputs["box_2d"]

        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.shape[0] == img.shape[0]
        assert heatmap.shape[1] == NUM_CLASSES
        assert heatmap.shape[2] == img.shape[2] // model.output_stride
        assert heatmap.shape[3] == img.shape[3] // model.output_stride

        assert isinstance(box_2d, torch.Tensor)
        assert box_2d.shape[0] == img.shape[0]
        assert box_2d.shape[1] == 4
        assert box_2d.shape[2] == img.shape[2] // model.output_stride
        assert box_2d.shape[3] == img.shape[3] // model.output_stride

    def test_forward_pass(self, backbone, neck, output_heads, task, sample_batch):
        img = sample_batch["image"]
        model = CenterNet(backbone, neck, output_heads, task)
        heatmap, box_2d = model(img)

        assert isinstance(heatmap, torch.Tensor)
        assert heatmap.max() <= 1
        assert heatmap.min() >= 0
        assert heatmap.shape[1] == NUM_CLASSES

        assert isinstance(box_2d, torch.Tensor)
        assert box_2d.shape[1] == 4

    def test_compute_loss(self, backbone, neck, output_heads, task, sample_batch, sample_model_outputs):
        model = CenterNet(backbone, neck, output_heads, task)
        losses = model.compute_loss(sample_model_outputs, sample_batch)

        # correct loss names and loss is not nan
        for x in ["heatmap", "box_2d", "total"]:
            assert isinstance(losses[x], torch.Tensor)
            assert not torch.isnan(losses[x])
