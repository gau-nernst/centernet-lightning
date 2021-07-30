import sys
import os
import tempfile
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def target_detections_to_coco_format(bboxes, labels, img_names=None, img_widths=None, img_heights=None, label_to_name=None):
    """Convert ground-truth detections (bboxes and labels) to COCO annotations format. Box must be in COCO format (xywh).
    
    Return a dict ready to be converted to JSON.
    """
    n_images = len(labels)
    img_names = range(n_images) if img_names is None else img_names
    img_widths = [1]*n_images if img_widths is None else img_widths
    img_heights = [1]*n_images if img_heights is None else img_heights

    images = []
    annotations = []
    label_set = set()

    # loop over images
    for i, (name, width, height, img_boxes, img_labels) in enumerate(zip(img_names, img_widths, img_heights, bboxes, labels)):
        image_info = {
            "id": i,
            "width": int(width),
            "height": int(height),
            "file_name": name
        }
        images.append(image_info)

        # loop over detections in an image
        for box, label in zip(img_boxes, img_labels):
            label_set.add(int(label))
            image_ann = {
                "id": len(annotations),
                "image_id": i,
                "category_id": int(label),
                "bbox": [float(x) for x in box],
                "area": float(box[2] * box[3]),
                "iscrowd": 0
            }
            annotations.append(image_ann)
    
    if label_to_name is None:
        categories = [{"id": label, "name": label} for label in label_set]
    else:
        categories = [{"id": label, "name": name} for label, name in label_to_name.items()]

    ann = {
        "info": None,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": None,
    }
    
    return ann

def pred_detections_to_coco_format(bboxes, labels, scores, image_ids=None, score_threshold=0.01):
    """Convert predicted detections (bboxes, labels, and scores) to COCO results format. Box must be in COCO format (xywh)
    
    Return a dict ready to be converted to JSON.
    """
    image_ids = range(len(labels)) if image_ids is None else image_ids

    results = []

    for img_id, img_bboxes, img_labels, img_scores in zip(image_ids, bboxes, labels, scores):
        for box, label, score in zip(img_bboxes, img_labels, img_scores):
            if score < score_threshold:
                continue

            item = {
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x) for x in box],
                "score": float(score)
            }
            results.append(item)

    return results

def evaluate_coco_detection(pred_bboxes, pred_labels, pred_scores, target_bboxes, target_labels, cat_ids=None, metrics_to_return=None):
    """Evaluate detections using COCO metrics from pycocotools
    """
    ann_file = None
    results_file = None
    
    ann_content = target_detections_to_coco_format(target_bboxes, target_labels)
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        ann_file = f.name
        json.dump(ann_content, f)
    
    results_content = pred_detections_to_coco_format(pred_bboxes, pred_labels, pred_scores)
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        results_file = f.name
        json.dump(results_content, f)

    return evaluate_coco_detection_from_file(ann_file, results_file, cat_ids=cat_ids, metrics_to_return=metrics_to_return)    

def evaluate_coco_detection_from_file(ann_file, results_file, cat_ids=None, metrics_to_return=None):
    metrics = (
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
    )
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")  # redirect pycocotools print() to null

    coco_gt = COCO(ann_file)
    coco_pred = coco_gt.loadRes(results_file)
    img_ids = sorted(coco_gt.getImgIds())

    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.imgIds = img_ids
    if cat_ids is not None:
        coco_eval.params.catIds = cat_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    sys.stdout = old_stdout     # restore print

    stats = {metric: coco_eval.stats[i] for i, metric in enumerate(metrics)}
    if metrics_to_return is not None:
        stats = {metric: stats[metric] for metric in metrics_to_return}
    
    return stats
