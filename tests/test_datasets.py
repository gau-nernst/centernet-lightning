import os
import random
import torch
from torch.utils.data import DataLoader
from src.datasets import COCODataset, VOCDataset, CollateDetectionsCenterNet, build_dataset, build_dataloader
from src.datasets.coco import prepare_coco_detection

COCO_DIR = "datasets/COCO"
VOC_DIR = "datasets/VOC2012"

class TestDatasets:
    def test_prepare_coco_detection(self):
        ann_dir = os.path.join(COCO_DIR, "annotations")
        prepare_coco_detection(ann_dir, "val2017", overwrite=True)

        assert os.path.exists(os.path.join(ann_dir, "detections_val2017.pkl"))
        assert os.path.exists(os.path.join(ann_dir, "label_to_name_val2017.json"))
        assert os.path.exists(os.path.join(ann_dir, "label_to_id_val2017.json"))

    def _test_dataset(self, dataset):
        rand_idx = random.randint(0, len(dataset)-1)
        sample = dataset[rand_idx]
        for x in ["image", "bboxes", "labels"]:
            assert x in sample

        img = sample["image"]
        assert isinstance(img, torch.Tensor)    # torch tensor
        assert img.shape == (3,512,512)         # correct default size
        assert torch.max(torch.abs(img)) < 3    # imagenet normalization

        bboxes = sample["bboxes"]
        for box in bboxes:
            assert len(box) == 4
            assert 0 < box[0] <= 1
            assert 0 < box[1] <= 1
            assert 0 < box[2] <= 1
            assert 0 < box[3] <= 1
        
        labels = sample["labels"]
        assert len(bboxes) == len(labels)

    def test_coco_dataset(self):
        ds = COCODataset(COCO_DIR, "val2017")
        self._test_dataset(ds)

    def test_voc_dataset(self):
        ds = VOCDataset(VOC_DIR, "val")
        self._test_dataset(ds)

class TestDataloader:
    def test_collate_fn(self):
        # should test this function alone without using COCODataset
        ds = COCODataset(COCO_DIR, "val2017")
        collate_fn = CollateDetectionsCenterNet((80,128,128))

        coco_dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(coco_dataloader))

        for x in ["image", "bboxes", "labels", "mask", "heatmap"]:
            assert x in batch
            assert isinstance(batch[x], torch.Tensor)

        assert batch["image"].shape == (4,3,512,512)
        assert batch["heatmap"].shape == (4,80,128,128)
