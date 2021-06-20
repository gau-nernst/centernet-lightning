import torch
from losses import *

class TestLosses:
    def test_render_target_heatmap(self):
        bboxes = torch.tensor([[
            [64,64,100,200],
            [64,80,50,100],
            [80,70,100,100]
        ]])
        labels = torch.tensor([[1,0,2]])
        mask = torch.tensor([[1,1,0]])

        heatmap = render_target_heatmap_ttfnet((1,3,128,128), bboxes, labels, mask)

        assert heatmap[0, labels[0][0], bboxes[0][0][1], bboxes[0][0][0]] == 1    # peak is 1
        assert torch.sum(heatmap == 1) == mask.sum()           # correct number of peaks

        heatmap = render_target_heatmap_cornernet((1,3,128,128), bboxes, labels, mask)

        assert heatmap[0, labels[0][0], bboxes[0][0][1], bboxes[0][0][0]] == 1    # peak is 1
        assert torch.sum(heatmap == 1) == mask.sum()           # correct number of peaks

    def test_focal_loss(self):
        sample_output = torch.rand((1,3,128,128))*10 - 5
        output_probs = torch.sigmoid(sample_output)

        bboxes = torch.tensor([[
            [64,64,100,200],
            [64,80,50,100],
            [80,70,100,100]
        ]])
        labels = torch.tensor([[1,0,2]])
        mask = torch.tensor([[1,1,0]])

        sample_target = render_target_heatmap_ttfnet((1,3,128,128), bboxes, labels, mask)
        focal_loss = FocalLossWithLogits(alpha=2, beta=4)

        loss1 = focal_loss(sample_output, sample_target)
        loss2 = reference_focal_loss(output_probs, sample_target)
        assert torch.abs(loss1 - loss2) < 1e-3
