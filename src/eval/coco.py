from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco(ann_file, results_file, cat_ids):
    metrics = (
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
    )

    coco_gt = COCO(ann_file)
    coco_pred = coco_gt.loadRes(results_file)
    img_ids = sorted(coco_gt.getImgIds())
    
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.params.catIds = cat_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = {metric: coco_eval.stats[i] for i, metric in enumerate(metrics)}
    return stats
