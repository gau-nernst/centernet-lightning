from typing import List
import warnings
from functools import partial

import torch
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from ..utils import box_iou_distance_matrix, box_giou_distance_matrix

torch.backends.cudnn.benchmark = True

class MatchingCost:
    _box_costs = {
        "iou": box_iou_distance_matrix,
        "giou": box_giou_distance_matrix
    }
    def __init__(self, reid_cost="cosine", box_cost="giou", reid_weight=1, box_weight=0):
        self.reid_cost_fn = partial(distance.cdist, metric=reid_cost)
        self.box_cost_fn = self._box_costs[box_cost]
        self.reid_weight = reid_weight
        self.box_weight = box_weight
    
    def __call__(self, reid_1, box_1, reid_2, box_2):
        reid_cost = self.reid_cost_fn(reid_1, reid_2)
        box_cost = self.box_cost_fn(box_1, box_2)

        cost = reid_cost * self.reid_weight + box_cost * self.box_weight
        return cost

class Tracker:
    # Tracktor: https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
    # DeepSORT: https://github.com/ZQPei/deep_sort_pytorch/blob/master/deep_sort/sort/tracker.py

    def __init__(self, model=None, device=None, nms_kernel=3, num_detections=300, detection_threshold=0.3, matching_threshold=0.2, matching_cost=None, smoothing_factor=0.5, use_kalman=False):
        self.model = model
        if model is None:
            warnings.warn("A model was not provided. Only `.update()` will work")
        
        # hparams
        self.device = getattr(model, "device", "cpu") if device is None else device
        self.nms_kernel = nms_kernel
        self.num_detections = num_detections
        self.detection_threshold = detection_threshold
        self.matching_threshold = matching_threshold
        self.matching_cost = matching_cost if matching_cost is not None else MatchingCost()
        self.smoothing_factor = smoothing_factor
        self.use_kalman = use_kalman

        # state variables
        self.frame = 0
        self.next_track_id = 0
        self.tracks: List[Track] = []

    def reset(self):
        self.frame = 0
        self.next_track_id = 0
        self.tracks = []

    @torch.no_grad()
    def step_batch(self, images: torch.Tensor, **kwargs):
        """

        Args:
            img: single image in CHW format
            kwargs: override post-processing config parameters e.g. nms_kernel, num_tracks

        Returns:
            a dict with keys "bboxes" and "track_ids"
        """
        device = kwargs.get("device", self.device)
        nms_kernel = kwargs.get("nms_kernel", self.nms_kernel)
        num_detections = kwargs.get("num_detections", self.num_detections)
        
        # forward pass
        self.model.eval()
        self.model.to(device)
        
        images = images.to(device)
        heatmap, box_2d, reid = self.model(images)

        # gather new detections and their embeddings
        new_detections = self.model.decode_tracking(heatmap, box_2d, reid, nms_kernel=nms_kernel, num_detections=num_detections, normalize_bbox=True)
        new_detections = {k: v.cpu().numpy() for k,v in new_detections.items()}

        out = {"bboxes": [], "track_ids": []}
        for b in range(images.shape[0]):
            bboxes = new_detections["bboxes"][b]
            labels = new_detections["labels"][b]
            scores = new_detections["scores"][b]
            embeddings = new_detections["embeddings"][b]
            self.update(bboxes, labels, scores, embeddings, **kwargs)    
            self.frame += 1

            track_bboxes = [x.bbox for x in self.tracks if x.active]
            track_ids = [x.track_id for x in self.tracks if x.active]
            out["bboxes"].append(track_bboxes)
            out["track_ids"].append(track_ids)
            
        return out

    @torch.no_grad()
    def step_single(self, img: torch.Tensor, **kwargs):
        img = img.unsqueeze(0)                  # add batch dim
        out = self.step_batch(img, **kwargs)
        out = {k: v[0] for k,v in out.items()}  # remove batch dim
        return out 

    def update(self, bboxes, labels, scores, embeddings, **kwargs):
        detection_threshold = kwargs.get("detection_threshold", self.detection_threshold)
        matching_threshold = kwargs.get("matching_threshold", self.matching_threshold)
        smoothing_factor = kwargs.get("smoothing_factor", self.smoothing_factor)
        use_kalman = kwargs.get("use_kalman", self.use_kalman)
        
        # filter by detection threshold
        mask = scores >= detection_threshold
        bboxes = bboxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        embeddings = embeddings[mask]

        if self.tracks:
            current_embeddings = np.stack([x.embedding for x in self.tracks], axis=0)
            current_bboxes = np.stack([x.bbox for x in self.tracks], axis=0)

            # embedding cost matrix
            cost_matrix = self.matching_cost(embeddings, bboxes, current_embeddings, current_bboxes)

            # match new detections with current active tracks
            # row is new detections, column is current tracks
            # TODO: filter by label first, then do matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            assigned_det = set()
            assigned_tracks = set()
        
            # only match if cost < threshold
            for row, col in zip(row_ind, col_ind):
                if cost_matrix[row, col] < matching_threshold:
                    self.tracks[col].update(bboxes[row], embeddings[row])
                    self.tracks[col].active = True
                    self.tracks[col].inactive_age = 0

                    assigned_det.add(row)
                    assigned_tracks.add(col)

            unmatched_detections = [x for x in range(len(bboxes)) if x not in assigned_det]
            unmatched_tracks = [x for x in range(len(self.tracks)) if x not in assigned_tracks]

        else:
            # initiate new tracks for all detections
            unmatched_detections = range(len(bboxes))
            unmatched_tracks = []

        # create new tracks from unmatched detections
        for idx in unmatched_detections:
            track = Track(self.next_track_id, bboxes[idx], labels[idx], embeddings[idx], smoothing_factor=smoothing_factor, use_kalman=use_kalman)
            self.tracks.append(track)
            self.next_track_id += 1
        
        # mark unmatched tracks as inactive
        for idx in unmatched_tracks:
            self.tracks[idx].active = False

        # increment inactive age, mark for delete, and predict next kalman state
        for track in self.tracks:
            track.step()

        # remove tracks
        self.tracks = [x for x in self.tracks if not x.to_delete]

class Track:
    """Track object
    """
    def __init__(self, track_id, bbox, label, embedding, smoothing_factor=0.9, max_inactive_age=30, use_kalman=False):
        """Initialize new track. Also initialize Kalman state and covariance if `use_kalman=True`
        """
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.embedding = embedding
        self.smoothing_factor = smoothing_factor
        self.max_inactivate_age = max_inactive_age

        self.active = True
        self.to_delete = False
        self.inactive_age = 0

        # kalman filter
        # https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
        self.kf = None
        if use_kalman:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.x = np.zeros(8)         # state vector
            self.kf.x[:4] = bbox            # first 4 values are box corners, last 4 values are velocities                 
            
            self.kf.F = np.eye(8)           # transition matrix. x_next = x_prev + v_prev dt (dt = 1)
            for i in range(4):              # [[1 1],
                self.kf.F[i, i+4] = 1       #  [0 1]]
            
            self.kf.H = np.eye(4,8)         # measurement matrix. only measure positions, velocities are not observed
            
            wh = bbox[2:] - bbox[:2]        # initiate covariance matrix as a diagonal matrix
            std = np.tile(wh, 4)            # values are relative to width/height (adapted from DeepSORT)
            std[:4] /= 10                   # std in position = wh/10
            std[4:] /= 16                   # std in velocity = wh/16
            self.kf.P = np.diag(std**2)

    def step(self):
        if not self.active:
            self.inactive_age += 1

            if self.inactive_age == self.max_inactivate_age:
                self.to_delete = True

        if self.kf is not None:
            wh = self.kf.x[2:4] - self.kf.x[:2]         # process noise relative to current width/height
            process_std = np.tile(wh, 4)
            process_std[:4] /= 20
            process_std[4:] /= 160
            process_noise = np.diag(process_std**2)
            self.kf.predict(Q=process_noise)

    def update(self, bbox, embedding):
        if self.kf is None:
            self.bbox = bbox
        
        else:
            wh = self.kf.x[2:4] - self.kf.x[:2]         # measurement noise relative to current width/height
            measure_std = np.tile(wh, 2)
            measure_std[:4] /= 20
            measure_std[4:] /= 160
            measure_noise = np.diag(measure_std**2)
            self.kf.update(bbox, R=measure_noise)
            self.bbox = self.kf.x[:4]

        self.embedding = (1-self.smoothing_factor) * self.embedding + self.smoothing_factor * embedding

    def __repr__(self):
        return f"track id: {self.track_id}, bbox: {self.bbox}, label: {self.label}, embedding: {len(self.embedding)} dim"
