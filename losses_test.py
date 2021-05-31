import torch
from losses import render_gaussian_kernel, FocalLossWithLogits, reference_focal_loss

class TestLosses:
    OUTPUT_SIZE = 128

    def test_render_gaussian_kernel(self):
        heatmap = torch.zeros((self.OUTPUT_SIZE,self.OUTPUT_SIZE))
        center_x = 64
        center_y = 64
        box_w = 10
        box_h = 20
        
        output_heatmap = render_gaussian_kernel(heatmap, center_x, center_y, box_w, box_h)
        assert output_heatmap[center_x, center_y] == 1
        assert output_heatmap[center_x+5, center_y+5] > 0
        assert torch.abs(output_heatmap[center_x+5, center_y] - output_heatmap[center_x-5, center_y]) < 1e-10

    def test_focal_loss(self):
        sample_output = torch.rand((10,self.OUTPUT_SIZE,self.OUTPUT_SIZE))
        output_probs = torch.sigmoid(sample_output)

        sample_target = [render_gaussian_kernel(torch.zeros(self.OUTPUT_SIZE,self.OUTPUT_SIZE),64,64,10,20) for _ in range(10)]
        sample_target = torch.stack(sample_target, dim=0)

        focal_loss = FocalLossWithLogits(alpha=2, beta=4)
        assert torch.abs(focal_loss(sample_output, sample_target) - reference_focal_loss(output_probs, sample_target)) < 1e-3
