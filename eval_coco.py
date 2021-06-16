import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CenterNet, fpn_resnet_backbone
from datasets import COCODataset, collate_detections_with_padding
from utils import convert_cxcywh_to_xywh
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_coco(checkpoint, force_inference=False):
    detection_file ="coco_val2017_results.json"

    if not os.path.exists("detection_file") or force_inference:
        print("Running inference on COCO val2017")

        print("Loading dataset")
        val_dataset = COCODataset("datasets/COCO", "val2017", eval=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers=4, collate_fn=collate_detections_with_padding)

        label_to_id = {v: k for k,v in val_dataset.id_to_label.items()}

        print("Creating and restoring model")
        backbone = fpn_resnet_backbone("resnet50")
        model = CenterNet.load_from_checkpoint(checkpoint, backbone=backbone, num_classes=val_dataset.num_classes)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = []
        
        print("Running inference")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                img_ids = batch["img_ids"].numpy()
                img_widths = batch["original_widths"].numpy()
                img_heights = batch["original_heights"].numpy()

                batch = {k: v.to(device) for k,v in batch.items()}
                encodings = model(batch)
                pred_detections = model.decode_detections(encodings, num_detections=100)

                pred_labels = pred_detections["labels"].cpu().numpy()
                pred_bboxes = pred_detections["bboxes"].cpu().numpy()
                pred_scores = pred_detections["scores"].cpu().numpy()

                for i in range(len(pred_labels)):
                    photo_img_id = img_ids[i]
                    photo_labels = pred_labels[i]
                    photo_bboxes = pred_bboxes[i]
                    photo_scores = pred_scores[i]

                    photo_bboxes[...,[0,2]] *= img_widths[i]
                    photo_bboxes[...,[1,3]] *= img_heights[i]
                    convert_cxcywh_to_xywh(photo_bboxes)

                    for label, bbox, score in zip(photo_labels, photo_bboxes, photo_scores):
                        item = {
                            "image_id": photo_img_id.item(),
                            "category_id": label_to_id[label.item()],
                            "bbox": bbox.tolist(),
                            "score": score.item()
                        }

                    results.append(item)

        print("Writing detections to file")
        with open(detection_file, "w", encoding="utf-8") as f:
            json.dump(results, f)

    print("Running COCO evaluation")
    coco_gt = COCO("datasets/COCO/annotations/instances_val2017.json")
    coco_pred = coco_gt.loadRes(detection_file)
    img_ids = sorted(coco_gt.getImgIds())
    
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.imgIds = img_ids
    # coco_eval.params.catIds = [1]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    eval_coco("wandb/run-20210614_121420-3mgn6vai/files/CenterNet/3mgn6vai/checkpoints/epoch=19-step=295719.ckpt", force_inference=True)