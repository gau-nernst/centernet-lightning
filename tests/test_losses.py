import numpy as np
import torch
from src.losses import FocalLossWithLogits
from src.losses.utils import reference_focal_loss
from src.datasets import render_target_heatmap_cornernet, render_target_heatmap_ttfnet

class TestLosses:
    def test_render_target_heatmap(self):
        heatmap = np.zeros((3,128,128))
        bboxes = [
            [64,64,100,200],
            [64,80,50,100],
            [80,70,100,100]
        ]
        labels = [1,0,2]
        bboxes_normalized = [[x/128 for x in box] for box in bboxes]

        heatmap = render_target_heatmap_ttfnet(heatmap, bboxes_normalized, labels)

        assert heatmap[labels[0], bboxes[0][1], bboxes[0][0]] == 1    # peak is 1
        assert np.sum(heatmap == 1) == len(labels)

        heatmap = render_target_heatmap_cornernet(heatmap, bboxes_normalized, labels)

        assert heatmap[labels[0], bboxes[0][1], bboxes[0][0]] == 1    # peak is 1
        assert np.sum(heatmap == 1) == len(labels)

    def test_focal_loss(self):
        # sample_output = torch.randn((4,3,128,128)) * 10 - 5
        # sample_logits = -torch.log((1-sample_output) / (sample_output+1e-8))

        # sample_target = torch.rand((4,3,128,128))
        # sample_target[0,0,64,64] = 1
        # sample_target[0,1,80,72] = 1

        # focal_loss = FocalLossWithLogits(alpha=2, beta=4)

        # loss1 = focal_loss(sample_logits, sample_target)
        # loss2 = reference_focal_loss(sample_output, sample_target)
        # assert torch.abs(loss1 - loss2) < 1e-3
