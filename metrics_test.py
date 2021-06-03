import numpy as np
from metrics import compute_iou_matrix, tpfp_detections, eval_detections

class TestMetrics:
    pred_bboxes = np.array([
        [10,20,20,40],
        [20,30,40,50],
        [10,10,20,20],
        [30,30,40,40],
        [4,10,40,20],
        [24,40,50,60]
    ]) / 128
    pred_scores = np.array([0.8, 0.2, 0.4, 0.9, 0.5, 0.2])
    pred_labels = np.array([0, 1, 0, 2, 1, 2])
    
    target_bboxes = np.array([
        [15,23,18,42],
        [10,30,15,40],
        [25,25,30,30],
        [30,30,40,40]
    ]) / 128
    target_labels = np.array([0, 0, 0, 2, 2, 2])

    num_classes = 4

    def test_compute_iou_matrix(self):
        bboxes1 = self.pred_bboxes
        bboxes2 = self.target_bboxes

        iou_matrix = compute_iou_matrix(bboxes1, bboxes2)

        assert iou_matrix.shape == (bboxes1.shape[0], bboxes2.shape[0])     # correct shape
        assert np.min(iou_matrix) >= 0          # all non-negative values
        assert np.max(iou_matrix) <= 1          # all values < 1
        assert not np.isnan(iou_matrix.sum())   # no nan

        assert iou_matrix[2,2] == 0     # test no overlap
        assert iou_matrix[3,3] == 1     # test full overlap

        # handle empty array
        iou_matrix = compute_iou_matrix(bboxes1, np.array([]))
        assert iou_matrix.shape[1] == 0
        iou_matrix = compute_iou_matrix(np.array([]), bboxes2)
        assert iou_matrix.shape[0] == 0

    def test_tpfp_detections(self):
        tp, fp = tpfp_detections(self.pred_bboxes, self.pred_scores, self.target_bboxes)
        
        assert tp + fp == self.pred_bboxes.shape[0]

        # handle 0 target detections
        tp, fp = tpfp_detections(self.pred_bboxes, self.pred_scores, np.array([]))
        assert tp == 0
        assert fp == self.pred_bboxes.shape[0]

        # handle 0 predicted detections
        tp, fp = tpfp_detections(np.array([]), np.array([]), self.target_bboxes)
        assert tp == 0
        assert fp == 0

    def test_eval_detections(self):
        detections = {
            "bboxes": np.expand_dims(self.pred_bboxes, 0),
            "labels": np.expand_dims(self.pred_labels, 0),
            "scores": np.expand_dims(self.pred_scores, 0)
        }
        targets = {
            "bboxes": np.expand_dims(self.target_bboxes, 0),
            "labels": np.expand_dims(self.target_labels, 0),
        }

        ap50, ar50 = eval_detections(detections, targets, self.num_classes)
