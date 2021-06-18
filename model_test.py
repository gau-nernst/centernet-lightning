import torch

from model import *
from losses import render_target_heatmap_ttfnet

sample_cfg = "configs/coco_resnet34.yaml"

sample_input = {
    "image": torch.rand((4,3,512,512)),
    "bboxes": torch.rand((4,10,4)) * 512,
    "labels": torch.randint(0,78,(4,10)),
    "mask": torch.randint(0,2,(4,10))
}

sample_output = {
    "heatmap": torch.rand((4,78,128,128)),
    "size": torch.rand((4,2,128,128)) * 128,
    "offset": torch.rand((4,2,128,128))
}

class TestModels:
    def test_build_centernet(self):
        model = build_centernet_from_cfg(sample_cfg)

    def test_forward_pass(self):
        model = build_centernet_from_cfg(sample_cfg)
        
        output = model(sample_input)
        
        for x in ["heatmap", "size", "offset"]:
            assert x in output
            assert not torch.isnan(torch.sum(output[x]))
        
        # correct output dimension
        assert output["heatmap"].shape == (4,81,128,128)
        assert output["size"].shape == (4,2,128,128)
        assert output["offset"].shape == (4,2,128,128)
    

    def test_compute_loss(self):
        model = build_centernet_from_cfg(sample_cfg)
        losses = model.compute_loss(sample_output, sample_input)

        # correct loss names and loss is not nan
        for x in ["heatmap", "size", "offset"]:
            assert x in losses
            assert not torch.isnan(losses[x])

    def test_decode_detections(self):
        centers = torch.tensor([[10,10], [20,30]])
        sizes = torch.tensor([[10,10], [10,20]])
        indices = torch.tensor([1,0])
        mask = torch.tensor([1,1])
        
        x1 = centers[0][0]
        y1 = centers[0][1]
        
        heatmap = render_target_heatmap_ttfnet((4,128,128), centers, sizes, indices, mask) * 0.95
        heatmap[indices[0],y1,x1] = 1                               # make the first point having highest score
        heatmap = -torch.log((1 - heatmap) / (heatmap + 1e-8))      # inverse sigmoid, convert probabilities to logits

        pred_size = torch.rand((1,2,128,128)) * 20
        pred_offset = torch.rand((1,2,128,128))

        sample_output = {
            "heatmap": heatmap.unsqueeze(0),
            "size": pred_size,
            "offset": pred_offset
        }
        model = build_centernet_from_cfg(sample_cfg)
        output = model.decode_detections(sample_output, num_detections=50)

        for x in ["labels", "bboxes", "scores"]:
            assert x in output
            assert output[x].shape[1] == 50

        labels = output["labels"].squeeze(dim=0)
        bboxes = output["bboxes"].squeeze(dim=0)
        scores = output["scores"].squeeze(dim=0)
        
        assert labels[0] == indices[0]
        assert scores[0] == 1

        assert bboxes[0][0] == x1 + pred_offset[0,0,y1,x1]
        assert bboxes[0][1] == y1 + pred_offset[0,1,y1,x1]
        assert bboxes[0][2] == pred_size[0,0,y1,x1]
        assert bboxes[0][3] == pred_size[0,1,y1,x1]
