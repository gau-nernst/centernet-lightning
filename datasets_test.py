import os
import random
import torch
from torch.utils.data import DataLoader
from datasets import COCODataset, prepare_coco_detection, collate_detections_with_padding

COCO_DIR = os.path.join("datasets", "COCO")

class TestDatasets:
    def test_prepare_coco_detection(self):
        prepare_coco_detection(COCO_DIR, "val2017")

        export_dir = os.path.join(COCO_DIR, "annotations", "val2017")
        assert os.path.exists(os.path.join(export_dir, "detections.pkl"))
        assert os.path.exists(os.path.join(export_dir, "label_to_name.json"))
        assert os.path.exists(os.path.join(export_dir, "label_to_id.json"))

    def test_coco_dataset(self):
        ds = COCODataset(COCO_DIR, "val2017")

        rand_idx = random.randint(0,len(ds)-1)
        sample = ds[rand_idx]
        for x in ["image", "bboxes", "labels"]:
            assert x in sample

        img = sample["image"]
        assert type(img) == torch.Tensor    # torch tensor
        assert img.shape == (3,512,512)     # check for correct default size
        assert torch.max(torch.abs(img)) <= 3

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
            assert label < 81

    def test_collate_fn(self):
        ds = COCODataset(COCO_DIR, "val2017")
        coco_dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_detections_with_padding)
        batch = next(iter(coco_dataloader))

        for x in ["image", "bboxes", "labels", "mask"]:
            assert x in batch
            assert type(batch[x]) == torch.Tensor

        assert batch["image"].shape == (4,3,512,512)
