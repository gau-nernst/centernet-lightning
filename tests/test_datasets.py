import os
import random

import yaml
import numpy as np
import torch

from centernet_lightning.datasets import COCODataset, VOCDataset, CrowdHumanDataset, MOTTrackingSequence, MOTTrackingDataset, KITTITrackingSequence, KITTITrackingDataset
from centernet_lightning.datasets.utils import get_default_detection_transforms, get_default_tracking_transforms, CollateDetection, CollateTracking
from centernet_lightning.datasets.builder import build_dataset, build_dataloader

COCO_DIR = "datasets/COCO"
VOC_DIR = "datasets/VOC2012"
CROWDHUMAN_DIR = "datasets/CrowdHuman"

MOT_TRACKING_DIR = "datasets/MOT/MOT17/train"
KITTI_TRACKING_DIR = "datasets/KITTI/tracking/training"

sample_detection_config = "configs/base_resnet34.yaml"
sample_tracking_config = "configs/base_tracking_resnet34.yaml"

def _test_detection_dataset(dataset, augmented=False):
    for _ in range(10):
        rand_idx = random.randint(0, len(dataset)-1)
        sample = dataset[rand_idx]
        for x in ["image", "bboxes", "labels"]:
            assert x in sample

        img = sample["image"]
        assert len(img.shape) == 3
        if augmented:
            assert isinstance(img, torch.Tensor)
            assert img.shape[0] == 3
            assert img.dtype == torch.float32
        
        else:
            assert isinstance(img, np.ndarray)
            assert img.shape[-1] == 3
            assert img.dtype == np.uint8

        bboxes = sample["bboxes"]
        for box in bboxes:
            assert len(box) == 4
            assert 0 < box[0] <= 1
            assert 0 < box[1] <= 1
            assert 0 < box[2] <= 1
            assert 0 < box[3] <= 1
        
        labels = sample["labels"]
        assert len(bboxes) == len(labels)

def _test_tracking_dataset(dataset, augmented=False):
    for _ in range(10):
        rand_idx = random.randint(0, len(dataset)-1)
        sample = dataset[rand_idx]
        for x in ["image", "bboxes", "labels", "ids"]:
            assert x in sample

        img = sample["image"]
        assert len(img.shape) == 3
        if augmented:
            assert isinstance(img, torch.Tensor)
            assert img.shape[0] == 3
            assert img.dtype == torch.float32
        
        else:
            assert isinstance(img, np.ndarray)
            assert img.shape[-1] == 3
            assert img.dtype == np.uint8

        bboxes = sample["bboxes"]
        for box in bboxes:
            assert len(box) == 4
            assert 0 <= box[0] <= 1
            assert 0 <= box[1] <= 1
            assert 0 < box[2] <= 1
            assert 0 < box[3] <= 1
        
        labels = sample["labels"]
        assert len(bboxes) == len(labels)

        ids = sample["ids"]
        assert len(ids) == len(labels)

class TestDetectionDatasets:
    def test_coco_dataset(self):
        ds = COCODataset(COCO_DIR, "val2017")
        _test_detection_dataset(ds, augmented=False)

        transforms = get_default_detection_transforms()
        ds = COCODataset(COCO_DIR, "val2017", transforms=transforms)
        _test_detection_dataset(ds, augmented=True)

    def test_voc_dataset(self):
        ds = VOCDataset(VOC_DIR, "val")
        _test_detection_dataset(ds, augmented=False)

        transforms = get_default_detection_transforms()
        ds = VOCDataset(VOC_DIR, "val", transforms=transforms)
        _test_detection_dataset(ds, augmented=True)

    def test_crowdhuman_dataset(self):
        ds = CrowdHumanDataset(CROWDHUMAN_DIR, "val")
        _test_detection_dataset(ds, augmented=False)

        transforms = get_default_detection_transforms()
        ds = CrowdHumanDataset(CROWDHUMAN_DIR, "val", transforms=transforms)
        _test_detection_dataset(ds, augmented=True)

class TestTrackingDatasets:
    def test_mot_dataset(self):
        names = os.listdir(MOT_TRACKING_DIR)
        sorted(names)

        ds = MOTTrackingSequence(MOT_TRACKING_DIR, names[0])
        _test_tracking_dataset(ds, augmented=False)
        ds = MOTTrackingDataset(MOT_TRACKING_DIR, names)
        _test_tracking_dataset(ds, augmented=False)

        transforms = get_default_tracking_transforms()
        ds = MOTTrackingSequence(MOT_TRACKING_DIR, names[0], transforms=transforms)
        _test_tracking_dataset(ds, augmented=True)
        ds = MOTTrackingDataset(MOT_TRACKING_DIR, names, transforms=transforms)
        _test_tracking_dataset(ds, augmented=True)
    
    def test_kitti_dataset(self):
        names = os.listdir(os.path.join(KITTI_TRACKING_DIR, "image_02"))
        sorted(names)

        ds = KITTITrackingSequence(KITTI_TRACKING_DIR, names[0])
        _test_tracking_dataset(ds, augmented=False)
        ds = KITTITrackingDataset(KITTI_TRACKING_DIR, names)
        _test_tracking_dataset(ds, augmented=False)

        transforms = get_default_tracking_transforms()
        ds = KITTITrackingSequence(KITTI_TRACKING_DIR, names[0], transforms=transforms)
        _test_tracking_dataset(ds, augmented=True)
        ds = KITTITrackingDataset(KITTI_TRACKING_DIR, names, transforms=transforms)
        _test_tracking_dataset(ds, augmented=True)

def _get_dataset_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config["data"]["train"]["dataset"]

def _get_dataloader_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config["data"]["train"]

class TestBuilder:
    def test_dataset_builder(self):
        config = _get_dataset_config(sample_detection_config)
        dataset = build_dataset(config)
        assert isinstance(dataset, COCODataset)

        config = _get_dataset_config(sample_tracking_config)
        dataset = build_dataset(config)
        assert isinstance(dataset, MOTTrackingDataset)

    def test_dataloader_builder(self):
        config = _get_dataloader_config(sample_detection_config)
        dataloader = build_dataloader(config)
        batch = next(iter(dataloader))

        for x in ["image", "bboxes", "labels", "mask"]:
            assert x in batch

        assert len(batch["image"].shape) == 4
        assert batch["bboxes"].shape[1] == batch["labels"].shape[1]
        assert batch["bboxes"].shape[1] == batch["mask"].shape[1]

        config = _get_dataloader_config(sample_tracking_config)
        dataloader = build_dataloader(config)
        batch = next(iter(dataloader))

        for x in ["image", "bboxes", "labels", "ids", "mask"]:
            assert x in batch

        assert len(batch["image"].shape) == 4
        assert batch["bboxes"].shape[1] == batch["labels"].shape[1]
        assert batch["bboxes"].shape[1] == batch["ids"].shape[1]
        assert batch["bboxes"].shape[1] == batch["mask"].shape[1]
        