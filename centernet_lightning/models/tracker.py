from typing import List
from enum import Enum, auto
import warnings
from functools import partial

import torch
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from ..utils import box_iou_distance_matrix, box_giou_distance_matrix, load_config

torch.backends.cudnn.benchmark = True

class TrackState(Enum):
    UNCONFIRMED = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    TO_DELETE = auto()

_box_costs = {
    "iou": box_iou_distance_matrix,
    "giou": box_giou_distance_matrix
}

def match_with_threshold(cost_matrix, threshold):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    matched_row = set()
    matched_col = set()

    # only match if cost < threshold
    for row, col in zip(row_ind, col_ind):
        if cost_matrix[row, col] < threshold:
            matches.append((row,col))
            matched_row.add(row)
            matched_col.add(col)

    unmatched_row = [x for x in range(cost_matrix.shape[0]) if x not in matched_row]
    unmatched_col = [x for x in range(cost_matrix.shape[1]) if x not in matched_col]

    return matches, unmatched_row, unmatched_col

class Tracker:
    """Perform Multiple Object Tracking
    """
    # Tracktor: https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
    # DeepSORT: https://github.com/ZQPei/deep_sort_pytorch/blob/master/deep_sort/sort/tracker.py

    def __init__(self, model=None, nms_kernel=3, num_detections=300, detection_threshold=0.3, reid_cost="cosine", reid_threshold=0.2, box_cost="iou", box_threshold=0.5, smoothing_factor=0.5, use_kalman=False, max_inactive_age=30, min_birth_age=2):
        self.model = model
        if model is None:
            warnings.warn("A model was not provided. Only `.update()` will work")
        
        # detection params
        self.nms_kernel = nms_kernel
        self.num_detections = num_detections
        self.detection_threshold = detection_threshold

        # matching params
        self.reid_cost = partial(distance.cdist, metric=reid_cost) if isinstance(reid_cost, str) else reid_cost
        self.reid_threshold = reid_threshold
        self.box_cost = _box_costs[box_cost] if isinstance(box_cost, str) else box_cost
        self.box_threshold = box_threshold
        
        # track params
        self.smoothing_factor = smoothing_factor
        self.use_kalman = use_kalman
        self.max_inactive_age = max_inactive_age
        self.min_birth_age = min_birth_age

        # tracker state variables
        self.frame = 0
        self.next_track_id = 0
        self.tracks: List[Track] = []

    def reset(self):
        self.frame = 0
        self.next_track_id = 0
        self.tracks = []

    @torch.no_grad()
    def step_batch(self, images: torch.Tensor, **kwargs):
        """Inference on a batch of images and update current tracks

        Args:
            images: single image in CHW format
            kwargs: override hyperparameters e.g. nms_kernel, num_tracks

        Returns:
            a dict with keys "bboxes" and "track_ids", containing tracks' bboxes and track_ids in each image frame
        """
        nms_kernel = kwargs.get("nms_kernel", self.nms_kernel)
        num_detections = kwargs.get("num_detections", self.num_detections)
        
        # forward pass
        self.model.eval()
        images = images.to(self.model.device)
        heatmap, box_2d, reid = self.model(images)

        # gather new detections and their embeddings
        new_detections = self.model.gather_tracking2d(
            heatmap, box_2d, reid, 
            nms_kernel=nms_kernel, num_detections=num_detections, normalize_bbox=True
        )
        new_detections = {k: v.cpu().numpy() for k,v in new_detections.items()}

        out = {"bboxes": [], "track_ids": []}

        # iterate over each image frame
        for bboxes, labels, scores, embeddings in zip(new_detections["bboxes"], new_detections["labels"], new_detections["scores"], new_detections["embeddings"]):
            self.update(bboxes, labels, scores, embeddings, **kwargs)    
            self.frame += 1

            track_bboxes = [x.bbox for x in self.tracks if x.active]
            track_ids = [x.track_id for x in self.tracks if x.active]
            out["bboxes"].append(track_bboxes)
            out["track_ids"].append(track_ids)
            
        return out

    @torch.no_grad()
    def step_single(self, img: torch.Tensor, **kwargs):
        """Inference on a single image and update current tracks
        """
        img = img.unsqueeze(0)                  # add batch dim
        out = self.step_batch(img, **kwargs)
        out = {k: v[0] for k,v in out.items()}  # remove batch dim
        return out 

    def update(self, bboxes, labels, scores, embeddings, **kwargs):
        """Update current tracks with new detections (bboxes, labels, scores, and embeddings)
        """
        detection_threshold = kwargs.get("detection_threshold", self.detection_threshold)
        reid_threshold = kwargs.get("reid_threshold", self.reid_threshold)
        box_threshold = kwargs.get("box_threshold", self.box_threshold)
        
        # filter by detection threshold
        mask = scores >= detection_threshold
        det_bboxes = bboxes[mask]
        det_labels = labels[mask]
        det_embeddings = embeddings[mask]

        # TODO: support for multi-class tracking?
        # filter by label first, then do matching
        # is it necessary?
        if len(self.tracks) == 0:
            # no existing tracks, initiate all detections as new tracks
            unmatched_dets = range(len(det_bboxes))

        else:
            track_embeddings = np.stack([x.embedding for x in self.tracks], axis=0)
            track_bboxes = np.stack([x.bbox for x in self.tracks], axis=0)

            # match by reid embeddings
            reid_cost_matrix = self.reid_cost(det_embeddings, track_embeddings)
            matches, unmatched_dets, unmatched_tracks = match_with_threshold(reid_cost_matrix, reid_threshold)

            # match by box iou
            if self.box_cost is not None:
                # to map new index back to original index
                det_idx_mapper = {i: x for i,x in enumerate(unmatched_dets)}
                track_idx_mapper = {i: x for i,x in enumerate(unmatched_tracks)}
                
                remain_det_bboxes = det_bboxes[unmatched_dets]
                remain_track_bboxes = track_bboxes[unmatched_tracks]

                box_cost_matrix = self.box_cost(remain_det_bboxes, remain_track_bboxes)
                new_matches, unmatched_dets, unmatched_tracks = match_with_threshold(box_cost_matrix, box_threshold)
                
                # map to original indices
                new_matches = [(det_idx_mapper[x], track_idx_mapper[y]) for (x,y) in new_matches]
                unmatched_dets = [det_idx_mapper[x] for x in unmatched_dets]
                unmatched_tracks = [track_idx_mapper[x] for x in unmatched_tracks]

                # combine matches and subtract unmatched dets and tracks
                matches.extend(new_matches)

            for (det_idx, track_idx) in matches:
                self.tracks[track_idx].update_matched(bboxes[det_idx], embeddings[det_idx])

            for track_idx in unmatched_tracks:
                self.tracks[track_idx].update_unmatched()

        # create new tracks from unmatched detections
        for det_idx in unmatched_dets:
            track = Track(
                self.next_track_id, det_bboxes[det_idx], det_labels[det_idx], det_embeddings[det_idx],
                min_birth_age=self.min_birth_age, max_inactive_age=self.max_inactive_age,
                smoothing_factor=self.smoothing_factor, use_kalman=self.use_kalman
            )
            self.tracks.append(track)
            self.next_track_id += 1

        # remove tracks
        self.tracks = [x for x in self.tracks if not x.to_delete]

        # kalman filter predict step
        for track in self.tracks:
            track.kalman_predict()

def xyxy_to_xyah(box):
    box = box.copy()
    box[2:] = box[2:] - box[:2]         # xywh
    box[:2] = box[:2] + box[2:] / 2     # cxcywh
    box[2] = box[2] / box[3]            # cxcyah
    return box

def xyah_to_xyxy(box):
    box = box.copy()
    box[2] = box[2] * box[3]            # cxcywh
    box[:2] = box[:2] - box[2:] / 2     # xywh
    box[2:] = box[:2] + box[2:]         # xyxy
    return box

class Track:
    """Track object
    """
    def __init__(self, track_id, bbox, label, embedding, min_birth_age=2, max_inactive_age=30, smoothing_factor=0.9, use_kalman=False):
        """Initialize new track. Also initialize Kalman state and covariance if `use_kalman=True`
        """
        # track info
        self.track_id = track_id
        self.state = TrackState.UNCONFIRMED
        self.birth_age = 0
        self.inactive_age = 0
        
        # track features
        self.bbox = bbox
        self.label = label
        self.embedding = embedding / np.linalg.norm(embedding)

        # params
        self.min_birth_age = min_birth_age          # track must be matched for x frames to be active
        self.max_inactivate_age = max_inactive_age  # track will be deleted after being inactive for x frames
        self.smoothing_factor = smoothing_factor    # exponential smoothing, smaller value = more smooth and more lag

        # kalman filter
        # https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
        self.kf = None
        if use_kalman:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)

            # state vector. first 4 values are box corners, last 4 values are velocities
            self.kf.x = np.zeros(8)
            self.kf.x[:4] = bbox
            
            # transition matrix. constant velocity model
            self.kf.F = np.eye(8)
            self.kf.F[:4, 4:] = np.eye(4)
            
            # measurement matrix. only positions are observed
            self.kf.H = np.eye(4,8)
            
            # initiate covariance matrix. it is a diagonal matrix
            wh = bbox[2:] - bbox[:2]
            std = np.tile(wh, 4)            # adapted from DeepSORT
            std[:4] /= 10                   # std in position = wh/10
            std[4:] /= 16                   # std in velocity = wh/16
            self.kf.P = np.diag(std**2)

            # xyah version
            # self.kf.x[:4] = xyxy_to_xyah(bbox)
            # h = bbox[3] - bbox[1]
            # std = [
            #     h/10, h/10, 1e-2, h/10,
            #     h/16, h/16, 1e-5, h/16
            # ]
            # self.kf.P = np.diag(np.square(std))

    @property
    def active(self):
        return self.state == TrackState.ACTIVE

    @property
    def confirmed(self):
        return self.state != TrackState.UNCONFIRMED

    @property
    def to_delete(self):
        return self.state == TrackState.TO_DELETE

    def kalman_predict(self):
        if self.kf is not None:
            # calculate process noise. adapted from DeepSORT
            wh = self.kf.x[2:4] - self.kf.x[:2]
            process_std = np.tile(wh, 4)
            process_std[:4] /= 20
            process_std[4:] /= 160
            process_noise = np.diag(np.square(process_std))
            self.kf.predict(Q=process_noise)

            # xyah version
            # h = self.kf.x[3]
            # process_std = [
            #     h/20, h/20, 1e-2, h/20,
            #     h/160, h/160, 1e-5, h/160
            # ]
            # process_noise = np.diag(np.square(process_std))
            # self.kf.predict(Q=process_noise)

    def update_matched(self, bbox, embedding):
        # state management
        if self.state == TrackState.UNCONFIRMED:
            self.birth_age += 1
            if self.birth_age >= self.min_birth_age:
                self.state = TrackState.ACTIVE

        elif self.state == TrackState.INACTIVE:
            self.state = TrackState.ACTIVE
            self.inactive_age = 0

        # update bbox with optional Kalman filter
        if self.kf is None:
            self.bbox = bbox
        
        else:
            # calculate measurement noise. adapted from DeepSORT
            wh = self.kf.x[2:4] - self.kf.x[:2]
            measure_std = np.tile(wh, 2) / 20
            measure_noise = np.diag(measure_std**2)
            self.kf.update(bbox, R=measure_noise)
            self.bbox = self.kf.x[:4]

            # xyah version
            # h = self.kf.x[3]
            # measure_std = [h/20, h/20, 0.1, h/20]
            # measure_noise = np.diag(np.square(measure_std))
            # self.kf.update(xyxy_to_xyah(bbox), R=measure_noise)
            # self.bbox = xyah_to_xyxy(self.kf.x[:4])

        # update embedding
        embedding = embedding / np.linalg.norm(embedding)
        self.embedding = (1-self.smoothing_factor) * self.embedding + self.smoothing_factor * embedding

    def update_unmatched(self):
        # state management
        if self.state == TrackState.UNCONFIRMED:
            self.state = TrackState.TO_DELETE

        elif self.state == TrackState.ACTIVE:
            self.state = TrackState.INACTIVE
            self.inactive_age = 0
        
        elif self.state == TrackState.INACTIVE:
            self.inactive_age += 1
            if self.inactive_age >= self.max_inactivate_age:
                self.state = TrackState.TO_DELETE

    def __repr__(self):
        return f"track id: {self.track_id}, bbox: {self.bbox}, label: {self.label}, embedding: {len(self.embedding)} dim"

def build_tracker(config, model=None):
    if isinstance(config, str):
        config = load_config(config)["tracker"]

    return Tracker(model=model, **config)
