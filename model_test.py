import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import ResNetBackbone, CenterNet
from datasets import COCODataset, collate_detections_with_padding
from train import get_train_augmentations


DATASET_ROOT = os.path.join("datasets")
COCO_DIR = os.path.join(DATASET_ROOT, "COCO")
ds = COCODataset(COCO_DIR, "val2017", transforms=get_train_augmentations())
coco_dataloader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate_detections_with_padding)

class TestModels:
    INPUT_SIZE = 512
    OUTPUT_SIZE = INPUT_SIZE//4
    OUTPUT_HEADS = ["heatmap", "size", "offset"]

    def test_resnet_backbone(self):
        backbone = ResNetBackbone()
        sample_input = torch.rand((4,3,self.INPUT_SIZE,self.INPUT_SIZE))
        sample_output = backbone(sample_input)

        assert sample_output.shape == (4,64,self.OUTPUT_SIZE,self.OUTPUT_SIZE)  # output dimension

    def test_output_head(self):
        pass

    def test_forward_pass(self):
        batch = next(iter(coco_dataloader))

        backbone = ResNetBackbone()
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

        backbone = ResNetBackbone()
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)

        losses = model.compute_loss(batch)

        # correct loss names and loss is not nan
        for x in self.OUTPUT_HEADS:
            assert x in losses
            assert not torch.isnan(losses[x])

    def test_trainer(self):
        # make sure pytorch lightning trainer can run
        backbone = ResNetBackbone()
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)
        
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus, fast_dev_run=2)
        trainer.fit(model, coco_dataloader, coco_dataloader)
