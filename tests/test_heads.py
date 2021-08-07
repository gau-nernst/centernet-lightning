from random import sample
from numpy.lib.arraysetops import isin
import torch

from centernet_lightning.models.heads import HeatmapHead, Box2DHead

sample_input = {
    "image": torch.rand((4,3,512,512)),
    "bboxes": torch.rand((4,10,4)),
    "labels": torch.randint(0,80,(4,10)),
    "mask": torch.randint(0,2,(4,10))
}

sample_output = {
    "heatmap": torch.rand((4,80,128,128)),
    "box_2d": torch.rand((4,4,128,128))
}

class TestHeatmapHead:
    def test_forward(self):
        head = HeatmapHead(64, 80)
        outputs = head(torch.rand((4,64,128,128)))
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (4,80,128,128)

    def test_render_heatmap(self):
        heatmap = torch.zeros(80,128,128)
        bboxes = sample_input["bboxes"][0]
        labels = sample_input["labels"][0]
        mask = sample_input["mask"][0]

        head = HeatmapHead(64, 80)
        head._render_target_heatmap(heatmap, bboxes, labels, mask)
        assert heatmap.max() <= 1
        assert heatmap.min() >= 0
        assert heatmap.sum() > mask.sum()
        
        for box, label, m in zip(bboxes, labels, mask):
            if m == 1:
                x_index = (box[0]*128).long()
                y_index = (box[1]*128).long()
                assert heatmap[label, y_index, x_index] == 1

    def test_compute_loss(self):
        head = HeatmapHead(64, 80)
        loss = head.compute_loss(sample_output, sample_input)
        assert not torch.isnan(loss)

class TestBox2DHead:
    def test_forward(self):
        head = Box2DHead(64)
        outputs = head(torch.rand((4,64,128,128)))
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (4,4,128,128)

    def test_compute_loss(self):
        head = Box2DHead(64)
        loss = head.compute_loss(sample_output, sample_input)
        assert not torch.isnan(loss)
