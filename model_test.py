import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import CenterNet, simple_resnet_backbone, simple_mobilenet_backbone
from datasets import COCODataset, collate_detections_with_padding
from train import get_train_augmentations
from losses import render_target_heatmap


DATASET_ROOT = os.path.join("datasets")
COCO_DIR = os.path.join(DATASET_ROOT, "COCO")
ds = COCODataset(COCO_DIR, "val2017", transforms=get_train_augmentations())
coco_dataloader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate_detections_with_padding)

class TestModels:
    INPUT_SIZE = 512
    OUTPUT_SIZE = INPUT_SIZE//4
    OUTPUT_HEADS = ["heatmap", "size", "offset"]

    def test_resnet_backbone(self):
        for resnet in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = simple_resnet_backbone(resnet)
            sample_input = torch.rand((4,3,self.INPUT_SIZE,self.INPUT_SIZE))
            sample_output = backbone(sample_input)

            assert sample_output.shape == (4,64,self.OUTPUT_SIZE,self.OUTPUT_SIZE)  # output dimension

    def test_mobilenet_backbone(self):
        for mobilenet in ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]:
            backbone = simple_mobilenet_backbone(mobilenet)
            sample_input = torch.rand((4,3,self.INPUT_SIZE,self.INPUT_SIZE))
            sample_output = backbone(sample_input)

            assert sample_output.shape == (4,64,self.OUTPUT_SIZE,self.OUTPUT_SIZE)  # output dimension

    def test_forward_pass(self):
        batch = next(iter(coco_dataloader))

        backbone = simple_resnet_backbone("resnet50")
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)
        
        sample_output = model(batch)
        
        for x in self.OUTPUT_HEADS:
            assert x in sample_output
        
        # correct output dimension
        assert sample_output["heatmap"].shape == (4,ds.num_classes,self.OUTPUT_SIZE,self.OUTPUT_SIZE)
        assert sample_output["size"].shape == (4,2,self.OUTPUT_SIZE,self.OUTPUT_SIZE)
        assert sample_output["offset"].shape == (4,2,self.OUTPUT_SIZE,self.OUTPUT_SIZE)

        # no nan in output
        for x in self.OUTPUT_HEADS:
            assert not torch.isnan(torch.sum(sample_output[x]))

    def test_compute_loss(self):
        batch = next(iter(coco_dataloader))

        backbone = simple_resnet_backbone("resnet50")
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)

        output_maps = model(batch)
        losses = model.compute_loss(output_maps, batch)

        # correct loss names and loss is not nan
        for x in self.OUTPUT_HEADS:
            assert x in losses
            assert not torch.isnan(losses[x])

    def test_trainer(self):
        # make sure pytorch lightning trainer can run
        backbone = simple_resnet_backbone("resnet50")
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)
        
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus, fast_dev_run=2)
        trainer.fit(model, coco_dataloader, coco_dataloader)

    def test_decode_detections(self):
        shape = (4, self.OUTPUT_SIZE, self.OUTPUT_SIZE)
        centers = torch.tensor([[10,10], [20,30]])
        sizes = torch.tensor([[10,10], [10,20]])
        indices = torch.tensor([1,0])
        mask = torch.tensor([1,1])
        
        x1 = centers[0][0]
        y1 = centers[0][1]
        
        heatmap = render_target_heatmap(shape, centers, sizes, indices, mask) * 0.95
        heatmap[indices[0],y1,x1] = 1                   # make the first point having highest score
        heatmap = -torch.log((1-heatmap) / heatmap)     # convert probability to logit (inverse sigmoid)
        
        size = torch.rand((1,2,self.OUTPUT_SIZE,self.OUTPUT_SIZE)) * 20
        offset = torch.rand((1,2,self.OUTPUT_SIZE,self.OUTPUT_SIZE))

        sample_input = {
            "heatmap": heatmap.unsqueeze(0),
            "size": size,
            "offset": offset
        }
        backbone = simple_resnet_backbone("resnet50")
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)
        
        output = model.decode_detections(sample_input)
        for x in ["labels", "bboxes", "scores"]:
            assert x in output
        labels = output["labels"].squeeze(dim=0)
        bboxes = output["bboxes"].squeeze(dim=0)
        scores = output["scores"].squeeze(dim=0)

        assert labels[0] == indices[0]
        assert scores[0] == 1

        assert bboxes[0][0] == (x1 + offset[0,0,y1,x1]) / self.OUTPUT_SIZE
        assert bboxes[0][1] == (y1 + offset[0,1,y1,x1]) / self.OUTPUT_SIZE
        assert bboxes[0][2] == size[0,0,y1,x1]
        assert bboxes[0][3] == size[0,1,y1,x1]