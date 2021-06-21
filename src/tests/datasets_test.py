import os
import random
import torch
from torch.utils.data import DataLoader
from ..datasets import *
from ..datasets.coco import prepare_coco_detection

COCO_DIR = "datasets/COCO"

class TestDatasets:
    def test_prepare_coco_detection(self):
        ann_dir = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
        save_dir = os.path.join(COCO_DIR, "val2017")
        prepare_coco_detection(ann_dir, save_dir, overwrite=True)

        assert os.path.exists(os.path.join(save_dir, "detections.pkl"))
        assert os.path.exists(os.path.join(save_dir, "label_to_name.json"))
        assert os.path.exists(os.path.join(save_dir, "label_to_id.json"))

    def test_coco_dataset(self):
        data_dir = os.path.join(COCO_DIR, "val2017")
        detection_file = os.path.join(data_dir, "detections.pkl")
        ds = COCODataset(data_dir, detection_file)

        rand_idx = random.randint(0, len(ds)-1)
        sample = ds[rand_idx]
        for x in ["image", "bboxes", "labels"]:
            assert x in sample

        img = sample["image"]
        assert isinstance(img, torch.Tensor)    # torch tensor
        assert img.shape == (3,512,512)         # correct default size
        assert torch.max(torch.abs(img)) <= 3   # imagenet normalization

        bboxes = sample["bboxes"]
        for box in bboxes:
            assert len(box) == 4
            assert 0 < box[0] <= 512
            assert 0 < box[1] <= 512
            assert 0 < box[2] <= 512
            assert 0 < box[3] <= 512
        
        labels = sample["labels"]
        assert len(bboxes) == len(labels)
        for label in labels:
            assert label < 80

    def test_collate_fn(self):
        data_dir = os.path.join(COCO_DIR, "val2017")
        detection_file = os.path.join(data_dir, "detections.pkl")
        ds = COCODataset(data_dir, detection_file)
        coco_dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_detections_with_padding)
        batch = next(iter(coco_dataloader))

        for x in ["image", "bboxes", "labels", "mask"]:
            assert x in batch
            assert isinstance(batch[x], torch.Tensor)

        assert batch["image"].shape == (4,3,512,512)
