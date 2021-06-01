import torch
from losses import FocalLossWithLogits, reference_focal_loss, render_target_heatmap

class TestLosses:
    OUTPUT_SIZE = 128
    NUM_CLASSES = 10
    HEATMAP_SHAPE = (NUM_CLASSES, OUTPUT_SIZE, OUTPUT_SIZE)

    def test_render_target_heatmap(self):
        centers = torch.tensor([[10,10], [20,30], [14,30]])
        sizes = torch.tensor([[10,10], [10,20], [30,30]])
        indices = torch.tensor([1,0,2])
        mask = torch.tensor([1,1,0])

        heatmap = render_target_heatmap(self.HEATMAP_SHAPE, centers, sizes, indices, mask)

        assert heatmap[indices,centers[:,1],centers[:,0]].sum() == mask.sum()    # peak is 1 if mask == 1
        assert torch.sum(heatmap == 1) == mask.sum()           # correct number of peaks

    def test_focal_loss(self):
        sample_output = torch.rand(self.HEATMAP_SHAPE)*10 - 5
        output_probs = torch.sigmoid(sample_output)

        centers = torch.tensor([[10,10], [20,30], [14,30]])
        sizes = torch.tensor([[10,10], [10,20], [30,30]])
        indices = torch.tensor([1,0,2])
        mask = torch.tensor([1,1,0])

        sample_target = render_target_heatmap(self.HEATMAP_SHAPE, centers, sizes, indices, mask)
        focal_loss = FocalLossWithLogits(alpha=2, beta=4)

        loss1 = focal_loss(sample_output, sample_target)
        loss2 = reference_focal_loss(output_probs, sample_target)
        assert torch.abs(loss1 - loss2) < 1e-3
