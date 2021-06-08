import numpy as np

# NOTE: these function operate on numpy arrays and don't use python data structures. in theory they can be sped with numba.jit

def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray, eps: float=1e-6):
    """Compute IoU of every pair of bboxes in `bboxes1` and `bboxes2` in an image. bboxes are in `x1y1x2y2` format. Both relative and absolute scales should work

    Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/bbox_overlaps.py
    """
    # bboxes1 = bboxes1.astype(np.float32)
    # bboxes2 = bboxes2.astype(np.float32)

    # create a matrix of bboxes2 against bboxes1
    iou_matrix = np.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=np.float32)

    # if either bboxes is empty, return empty ndarray
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return iou_matrix

    # (x2 - x1) * (y2 - y1)
    area1 = (bboxes1[:,2] - bboxes1[:,0]) * (bboxes1[:,3] - bboxes1[:,1])
    area2 = (bboxes2[:,2] - bboxes2[:,0]) * (bboxes2[:,3] - bboxes2[:,1])

    # iterate over bboxes in bboxes1
    for i in range(bboxes1.shape[0]):
        # intersection between bboxes1[i] and all other bboxes in bboxes2
        x_start = np.maximum(bboxes1[i,0], bboxes2[:,0])
        y_start = np.maximum(bboxes1[i,1], bboxes2[:,1])
        x_end = np.minimum(bboxes1[i,2], bboxes2[:,2])
        y_end = np.minimum(bboxes1[i,3], bboxes2[:,3])

        # intersection area, clip (x2-x1) and (y2-y1) to 0
        intersection = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        union = area1[i] + area2 - intersection

        iou_matrix[i,:] = intersection / (union + eps)
    
    return iou_matrix

def tpfp_detections(pred_bboxes: np.ndarray, scores: np.ndarray, target_bboxes: np.ndarray, threshold: float=0.5):
    """Count the number of true positives and false positives in an image

    Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/mean_ap.py#L153
    """
    # tp = np.zeros(pred_bboxes.shape[0], dtype=np.float32)
    # fp = np.zeros(pred_bboxes.shape[0], dtype=np.float32)
    tp = 0
    fp = 0

    # no target bboxes, set all predicted bboxes as false positive
    if target_bboxes.shape[0] == 0:
        # fp[:] = 1
        fp = pred_bboxes.shape[0]
        return tp, fp
    
    iou_matrix = compute_iou_matrix(pred_bboxes, target_bboxes)

    # greedy match every predicted bbox with the best ground truth bbox
    iou_argmax = np.argmax(iou_matrix, axis=1)      # index of best matched target bbox for each predicted bbox
    iou_max = np.max(iou_matrix, axis=1)            # best iou value for each predicted bbox

    target_matched = np.zeros(target_bboxes.shape[0], dtype=bool)   # keep track which target bbox is already matched
    
    # consider predicted bboxes in descending order of their scores
    sort_indices = np.argsort(-scores)
    for i in sort_indices:
        # tp if iou >= threshold and target bbox is not yet matched, else fp (PASCAL VOC)
        if iou_max[i] >= threshold and not target_matched[iou_argmax[i]]:
            # tp[i] = 1
            tp += 1
            target_matched[iou_argmax[i]] = True
        else:
            # fp[i] = 1
            fp += 1
    
    return tp, fp

def class_tpfp_batch(detections, targets, num_classes, iou_threshold=0.5, detection_threshold=0.5, eps=1e-6):
    """Compute TP and FP on a batch of predicted and ground truth detections
    """
    # N x K predicted detections
    pred_bboxes = detections["bboxes"]
    pred_scores = detections["scores"]
    pred_labels = detections["labels"]

    # N x D groud truth detections
    target_bboxes = targets["bboxes"]
    target_labels = targets["labels"]
    target_mask = targets["mask"].astype(bool) if "mask" in targets else np.ones_like(target_labels)

    class_tp = np.zeros(num_classes, dtype=np.float32)
    class_fp = np.zeros(num_classes, dtype=np.float32)

    batch_size = pred_bboxes.shape[0]

    # NOTE: if use numba, can convert this to prange
    for b in range(batch_size):
        # first use target mask to remove non-detections from padding
        batch_target_bboxes = target_bboxes[b, target_mask[b], :]
        batch_target_labels = target_labels[b, target_mask[b]]

        # iterate over classes
        for i in range(num_classes):
            pred_class_indices = (pred_labels[b] == i) & (pred_scores[b] >= detection_threshold)
            target_class_indices = (batch_target_labels == i)

            batch_pred_bboxes_i = pred_bboxes[b,pred_class_indices,:]
            batch_pred_scores_i = pred_scores[b,pred_class_indices]
            batch_target_bboxes_i = batch_target_bboxes[target_class_indices,:]
            
            tp, fp = tpfp_detections(batch_pred_bboxes_i, batch_pred_scores_i, batch_target_bboxes_i, threshold=iou_threshold)
            class_tp[i] += tp
            class_fp[i] += fp
            
    return class_tp, class_fp
