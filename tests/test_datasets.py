import random

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from centernet_lightning.datasets import COCODataset, VOCDataset, CrowdHumanDataset, MOTTrackingSequence, MOTTrackingDataset, KITTITrackingSequence, KITTITrackingDataset
from centernet_lightning.datasets.utils import get_default_detection_transforms, get_default_tracking_transforms, CollateDetection, CollateTracking
from centernet_lightning.datasets.builder import build_dataset, build_dataloader

def generate_detection_dataset_configs():
    pass

def generate_tracking_dataset_configs():
    pass

class TestDetectionDataset:
    dataset_configs = generate_detection_dataset_configs()

    def test_attributes(self, constructor, data_dir, split, name_to_label):
        dataset = constructor(data_dir, split, name_to_label)

        assert isinstance(len(dataset), int)

    def test_get_item(self, constructor, data_dir, split, name_to_label):
        dataset = constructor(data_dir, split, name_to_label)
        
        for item in random.sample(dataset, 10):
            assert isinstance(item["image"], np.ndarray)
            assert item["image"].shape[-1] == 3

            assert isinstance(item["bboxes"], list)
            for box in item["bboxes"]:
                assert len(box) == 4
                for x in box:
                    assert 0 <= x <= 1
            
            assert isinstance(item["labels"], list)
            assert len(item["labels"]) == len(item["bboxes"])

        transforms = get_default_detection_transforms()
        dataset = constructor(data_dir, split, name_to_label, transforms=transforms)
        
        for item in random.sample(dataset, 10):
            assert isinstance(item["image"], torch.Tensor)
            assert item["image"].shape[0] == 3

            assert isinstance(item["bboxes"], tuple)
            assert isinstance(item["labels"], tuple)
            assert len(item["bboxes"]) == len(item["labels"])

    def test_dataloader(self, constructor, data_dir, split, name_to_label):
        batch_size = 4
        transforms = get_default_detection_transforms()
        collate_fn = CollateDetection()
        
        dataset = constructor(data_dir, split, name_to_label, transforms=transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dataloader))

        img = batch["image"]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == batch_size
        
        bboxes = batch["bboxes"]
        assert isinstance(bboxes, torch.Tensor)
        assert bboxes.shape[0] == batch_size
        assert bboxes.max() <= 1
        assert bboxes.min() >= 0

        labels = batch["labels"]
        assert isinstance(labels, torch.Tensor)
        assert labels.shape[0] == batch_size

        mask = batch["mask"]
        assert isinstance(mask, torch.Tensor)
        assert mask.shape[0] == batch_size
        for x in mask.view(-1):
            assert x == 0 or x == 1

        assert bboxes.shape[1] == labels.shape[1] == mask.shape[1]

    def test_builder(self):
        pass
class TestTrackingDataset:
    dataset_configs = generate_tracking_dataset_configs()

    def test_attributes(self):
        pass

    def test_get_item(self):
        pass

    def test_dataloader(self):
        pass

    def test_builder(self):
        pass