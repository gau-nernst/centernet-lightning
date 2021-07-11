import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import COCODataset, VOCDataset, CrowdHumanDataset, MOTTrackingSequence, KITTITrackingSequence
from src.datasets.utils import get_default_detection_transforms, get_default_tracking_transforms, CollateDetection, CollateTracking
from src.datasets.builder import build_dataset, build_dataloader

COCO_DIR = "datasets/COCO"
VOC_DIR = "datasets/VOC2012"
CROWDHUMAN_DIR = "datasets/CrowdHuman"

MOT_TRACKING_DIR = "datasets/MOT/MOT17/train"
KITTI_TRACKING_DIR = "datasets/KITTI/tracking/training"

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
            assert 0 < box[0] <= 1
            assert 0 < box[1] <= 1
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
        ds = MOTTrackingSequence(MOT_TRACKING_DIR, "MOT17-02-FRCNN")
        _test_tracking_dataset(ds, augmented=False)

        transforms = get_default_tracking_transforms()
        ds = MOTTrackingSequence(MOT_TRACKING_DIR, "MOT17-02-FRCNN", transforms=transforms)
        _test_tracking_dataset(ds, augmented=True)
    
    def test_kitti_dataset(self):
        ds = KITTITrackingSequence(KITTI_TRACKING_DIR, "0000")
        _test_tracking_dataset(ds, augmented=False)

        transforms = get_default_tracking_transforms()
        ds = KITTITrackingSequence(KITTI_TRACKING_DIR, "0000", transforms=transforms)
        _test_tracking_dataset(ds, augmented=True)

    