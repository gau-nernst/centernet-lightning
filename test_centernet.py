import os

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from losses import FocalLossWithLogits
# from ..CenterTrack.src.lib.model.losses import _neg_loss
from datasets import COCODataset

class TestDatasets:
    dataset_root = os.path.join(os.environ["HOME"], "thien", "datasets")
    
    def test_coco(self):
        coco_dir = os.path.join(self.dataset_root, "COCO")
        ds = COCODataset(coco_dir, "val2017")

        sample = ds[0]
        print(sample["img"].shape)
        print(sample["img"])
        assert "img" in sample
        assert "bboxes" in sample

        img = sample["img"]
        assert type(img) == torch.Tensor    # torch tensor
        assert len(img.shape) == 3          # dim = 3
        assert img.shape[0] == 3            # channels = 3, CHW
        assert torch.max(img) <= 1          # normalize to [0,1]

        bboxes = sample["bboxes"]
    
    def test_augmentation(self):
        coco_dir = os.path.join(self.dataset_root, "COCO")
        train_augment = A.Compose([
            A.HorizontalFlip(),
            A.RandomScale(),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
        ds = COCODataset(coco_dir, "val2017")

        sample1 = ds[0]
        sample2 = ds[1]
        diff = sample1["img"] - sample2["img"]
        assert (diff*diff).sum() > 1e-10        # due to augmentations, they should be 2 different images

class TestModels:
    def test_resnet_backbone(self):
        pass

    def test_centernet(self):
        pass

class TestUtils:
    def test_render_gaussian(self):
        pass

    def test_decode(self):
        pass

class TestLosses:
    def test_focal_loss(self):
        # focal_loss = FocalLossWithLogits(alpha=2, beta=4)
        # input1 = torch.random()
        # input1_probs = F.sigmoid(input1)
        pass
        
    def test_reg_l1_loss(self):
        pass