import os
import torch

from src.models import CenterNet, build_centernet_from_cfg
from src.backbones.simple import SimpleBackbone
from src.datasets import render_target_heatmap_cornernet

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
        assert isinstance(model, CenterNet)
        assert isinstance(model.backbone, SimpleBackbone)

        model(sample_input)
    
    # def test_build_all_configs(self):
    #     configs = os.listdir("configs")
    #     configs = [os.path.join("configs", cfg) for cfg in configs]
    #     for cfg in configs:
    #         model = build_centernet_from_cfg(cfg)
    #         model(sample_input)

    def test_forward_pass(self):
        model = build_centernet_from_cfg(sample_cfg)
        output = model(sample_input)
        
        for x in ["heatmap", "size", "offset"]:
            assert x in output
            assert not torch.isnan(torch.sum(output[x]))
        
        # correct output dimension
        assert output["heatmap"].shape == (4,80,128,128)
        assert output["size"].shape == (4,2,128,128)
        assert output["offset"].shape == (4,2,128,128)
    

    # def test_compute_loss(self):
    #     model = build_centernet_from_cfg(sample_cfg)
    #     losses = model.compute_loss(sample_output, sample_input)

    #     # correct loss names and loss is not nan
    #     for x in ["heatmap", "size", "offset"]:
    #         assert x in losses
    #         assert not torch.isnan(losses[x])

    # def test_decode_detections(self):
    #     bboxes = torch.tensor([[
    #         [64,64,100,200],
    #         [64,80,50,100],
    #         [80,70,100,100]
    #     ]])
    #     labels = torch.tensor([[1,0,2]])
    #     mask = torch.tensor([[1,1,0]])
        
    #     x1 = bboxes[0][0][0]
    #     y1 = bboxes[0][0][1]
        
    #     heatmap = render_target_heatmap_cornernet((1,4,128,128), bboxes, labels, mask) * 0.95
    #     heatmap[0, labels[0][0], y1, x1] = 1                           # make the first point having highest score
    #     heatmap = -torch.log((1 - heatmap) / (heatmap + 1e-8))      # inverse sigmoid, convert probabilities to logits

    #     pred_size = torch.rand((1,2,128,128)) * 20
    #     pred_offset = torch.rand((1,2,128,128))

    #     sample_output = {
    #         "heatmap": heatmap,
    #         "size": pred_size,
    #         "offset": pred_offset
    #     }
    #     model = build_centernet_from_cfg(sample_cfg)
    #     output = model.decode_detections(sample_output, num_detections=50)

    #     for x in ["labels", "bboxes", "scores"]:
    #         assert x in output
    #         assert output[x].shape[1] == 50

    #     out_label = output["labels"][0][0]
    #     out_score = output["scores"][0][0]
    #     out_bbox  = output["bboxes"][0][0]

    #     assert out_label == labels[0][0]
    #     assert out_score == 1

    #     assert out_bbox[0] == (x1 + pred_offset[0,0,y1,x1]) * model.output_stride
    #     assert out_bbox[1] == (y1 + pred_offset[0,1,y1,x1]) * model.output_stride
    #     assert out_bbox[2] == pred_size[0,0,y1,x1]
    #     assert out_bbox[3] == pred_size[0,1,y1,x1]
