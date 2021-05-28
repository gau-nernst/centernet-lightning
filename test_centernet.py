import os
import random

import torch
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader
from train import get_train_augmentations
from losses import FocalLossWithLogits, reference_focal_loss, render_gaussian_kernel
from datasets import COCODataset, collate_bbox_labels
from model import OutputHead, ResNetBackbone, CenterNet

class TestDatasets:
    dataset_root = os.path.join(os.environ["HOME"], "thien", "datasets")
    coco_dir = os.path.join(dataset_root, "COCO")

    def test_coco(self):
        ds = COCODataset(self.coco_dir, "val2017")

        rand_idx = random.randint(0,len(ds)-1)
        sample = ds[rand_idx]
        for x in ["image", "bboxes", "labels"]:
            assert x in sample

        img = sample["image"]
        assert type(img) == torch.Tensor    # torch tensor
        assert len(img.shape) == 3          # rank 3 tensor
        assert img.shape == (3,512,512)     # check for correct default size
        assert torch.max(img) <= 1          # normalize to [0,1]

        bboxes = sample["bboxes"]
        labels = sample["labels"]
        assert len(bboxes) == len(labels)

    def test_augmentation(self):
        train_augment = get_train_augmentations()
        ds = COCODataset(self.coco_dir, "val2017", transforms=train_augment)

        rand_idx = random.randint(0,len(ds)-1)
        sample1 = ds[rand_idx]["image"]
        sample2 = ds[rand_idx]["image"]
        diff = sample1 - sample2
        assert (diff*diff).sum() > 1e-10        # due to augmentations, they should be different
        assert sample1.shape == (3,512,512)     # size is still the same

    def test_collate_fn(self):
        ds = COCODataset(self.coco_dir, "val2017")
        coco_dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_bbox_labels)
        batch = next(iter(coco_dataloader))

        for x in ["image", "bboxes", "labels", "mask"]:
            assert x in batch
            assert type(batch[x]) == torch.Tensor

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
    
class TestModels:
    INPUT_SIZE = 512
    OUTPUT_SIZE = INPUT_SIZE/4
    coco_dir = os.path.join(os.environ["HOME"], "thien", "datasets", "COCO")
    
    def test_resnet_backbone(self):
        backbone = ResNetBackbone()

        # ds = COCODataset(self.coco_dir, "val2017")
        # coco_dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_bbox_labels)
        # batch = next(iter(coco_dataloader))
        
        sample_input = torch.rand((4,3,self.INPUT_SIZE,self.INPUT_SIZE))
        sample_output = backbone(sample_input)
        # sample_output = backbone(batch["image"])
        assert sample_output.shape == (4,64,self.OUTPUT_SIZE,self.OUTPUT_SIZE)  # output dimension

    def test_centernet(self):
        ds = COCODataset(self.coco_dir, "val2017")
        coco_dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_bbox_labels)
        batch = next(iter(coco_dataloader))

        # print(type(batch))
        # print(batch["image"].shape)
        backbone = ResNetBackbone()
        model = CenterNet(backbone=backbone, num_classes=ds.num_classes, batch_size=4)
        
        sample_output = model(**batch)
        
        for x in ["heatmap", "size", "offset"]:
            assert x in sample_output
        # correct output dimension
        assert sample_output["heatmap"].shape == (4,ds.num_classes,self.OUTPUT_SIZE,self.OUTPUT_SIZE)
        assert sample_output["size"].shape == (4,2,self.OUTPUT_SIZE,self.OUTPUT_SIZE)
        assert sample_output["offset"].shape == (4,2,self.OUTPUT_SIZE,self.OUTPUT_SIZE)

        # model.compute_loss(batch)