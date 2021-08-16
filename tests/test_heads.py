import pytest
import torch

from centernet_lightning.models.heads import HeatmapHead, Box2DHead, ReIDHead, build_output_heads

@pytest.fixture
def num_classes():
    return 20

@pytest.fixture
def input_channels():
    return 512

@pytest.fixture
def sample_batch(num_classes):
    batch_size = 4
    return {
        "image": torch.rand((batch_size,3,512,512)),
        "bboxes": torch.rand((batch_size,10,4)),
        "labels": torch.randint(0,num_classes,(batch_size,10)),
        "mask": torch.randint(0,2,(batch_size,10))
    }

@pytest.fixture
def sample_input(input_channels):
    return torch.rand((4,input_channels,128,128))

@pytest.fixture
def sample_output_dict():
    return {
        "heatmap": torch.rand((4,80,128,128)),
        "box_2d": torch.rand((4,4,128,128))
    }

class TestHeatmapHead:
    def test_forward(self, num_classes, input_channels, sample_input):
        head = HeatmapHead(input_channels, num_classes)
        output = head(sample_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == num_classes
        assert output.shape[2] == sample_input.shape[2]
        assert output.shape[3] == sample_input.shape[3]

    def test_render_heatmap(self, num_classes, input_channels, sample_input, sample_batch):
        height, width = sample_input.shape[-2:]
        bboxes = sample_batch["bboxes"]
        labels = sample_batch["labels"]
        mask = sample_batch["mask"]

        head = HeatmapHead(input_channels, num_classes)
        heatmap = head._render_target_heatmap(sample_input.shape, bboxes, labels, mask)
        
        assert heatmap.max() <= 1
        assert heatmap.min() >= 0
        assert heatmap.sum() > mask.sum()
        
        for hm_img, box_img, label_img, m_img in zip(heatmap, bboxes, labels, mask):
            for box, label, m in zip(box_img, label_img, m_img):
                if m == 1:
                    x_index = (box[0]*width).long()
                    y_index = (box[1]*height).long()
                    assert hm_img[label, y_index, x_index] == 1

    def test_compute_loss(self, num_classes, input_channels, sample_output_dict, sample_batch):
        head = HeatmapHead(input_channels, num_classes)
        loss = head.compute_loss(sample_output_dict, sample_batch)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

class TestBox2DHead:
    def test_forward(self, input_channels, sample_input):
        head = Box2DHead(input_channels)
        output = head(sample_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == 4
        assert output.shape[2] == sample_input.shape[2]
        assert output.shape[3] == sample_input.shape[3]

    def test_compute_loss(self, input_channels, sample_output_dict, sample_batch):
        head = Box2DHead(input_channels)
        loss = head.compute_loss(sample_output_dict, sample_batch)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

class TestReIDHead:
    pass
