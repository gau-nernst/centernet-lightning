import os, random
import torch
from torch.utils.data import DataLoader
from datasets import COCODataset, prepare_coco_detection, collate_detections_with_padding
from train import get_train_augmentations


DATASET_ROOT = os.path.join("datasets")
COCO_DIR = os.path.join(DATASET_ROOT, "COCO")

class TestDatasets:
    DATA_KEYS = ["image", "bboxes", "labels"]

    def test_prepare_coco_detection(self):
        prepare_coco_detection(COCO_DIR, "val2017")

    def test_coco(self):
        ds = COCODataset(COCO_DIR, "val2017")

        rand_idx = random.randint(0,len(ds)-1)
        sample = ds[rand_idx]
        for x in self.DATA_KEYS:
            assert x in sample

        img = sample["image"]
        assert type(img) == torch.Tensor    # torch tensor
        assert len(img.shape) == 3          # rank 3 tensor
        assert img.shape == (3,512,512)     # check for correct default size
        assert torch.max(img) <= 1          # normalize to [0,1]

        bboxes = sample["bboxes"]
        for box in bboxes:
            assert len(box) == 4
        
        labels = sample["labels"]
        assert len(bboxes) == len(labels)

    def test_augmentation(self):
        train_augment = get_train_augmentations()
        ds = COCODataset(COCO_DIR, "val2017", transforms=train_augment)

        rand_idx = random.randint(0,len(ds)-1)
        sample1 = ds[rand_idx]["image"]
        sample2 = ds[rand_idx]["image"]
        diff = sample1 - sample2
        assert (diff*diff).sum() > 1e-10        # due to augmentations, they should be different
        assert sample1.shape == (3,512,512)     # size is still the same

    def test_collate_fn(self):
        ds = COCODataset(COCO_DIR, "val2017")
        coco_dataloader = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate_detections_with_padding)
        batch = next(iter(coco_dataloader))

        for x in self.DATA_KEYS + ["mask"]:
            assert x in batch
            assert type(batch[x]) == torch.Tensor
