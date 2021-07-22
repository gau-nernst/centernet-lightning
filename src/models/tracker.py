import time
import os
from typing import List

import numpy as np
import torch
from torch import nn
from torchvision.ops import box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import cv2

from ..utils.image_annotate import draw_bboxes

torch.backends.cudnn.benchmark = True

def cosine_distance_matrix(v1: torch.Tensor, v2: torch.Tensor, eps=1e-6):
    """Calculate cosine distance matrix. When 1 vector is a zero vector, cosine distance with it will be 1
    Args
        v1: dim M x D
        v2: dim N x D

    Return
        v: dim M x N
    """
    v1_n = v1 / v1.norm(dim=-1, keepdim=True).clip(min=eps)
    v2_n = v2 / v2.norm(dim=-1, keepdim=True).clip(min=eps)

    cost_matrix = 1 - torch.matmul(v1_n, v2_n.T)
    return cost_matrix

class MatchingCost:
    _reid_costs = {
        "cosine": cosine_distance_matrix
    }
    _box_costs = {
        "iou": box_iou,
        "giou": generalized_box_iou
    }
    def __init__(self, reid_cost="cosine", box_cost="giou", reid_weight=1, box_weight=0):
        self.reid_cost_fn = self._reid_costs[reid_cost]
        self.box_cost_fn = self._box_costs[box_cost]
        self.reid_weight = reid_weight
        self.box_weight = box_weight
    
    def __call__(self, reid_1, box_1, reid_2, box_2):
        reid_cost = self.reid_cost_fn(reid_1, reid_2)
        box_cost = 1 - self.box_cost_fn(box_1, box_2)

        cost = reid_cost * self.reid_weight + box_cost * self.box_weight
        return cost

class Tracker:
    # Tracktor: https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
    # DeepSORT: https://github.com/ZQPei/deep_sort_pytorch/blob/master/deep_sort/sort/tracker.py

    def __init__(self, model: nn.Module, device="cpu", nms_kernel=3, num_detections=100, detection_threshold=0.1, matching_threshold=0.2, matching_cost=None, smoothing_factor=0.9, transforms=None):
        model.eval()
        self.model = model
        self.device = device
        self.nms_kernel = nms_kernel
        self.num_detections = num_detections
        self.detection_threshold = detection_threshold
        self.matching_threshold = matching_threshold
        self.matching_cost = matching_cost if matching_cost is not None else MatchingCost(reid_weight=1, box_weight=0)
        self.smoothing_factor = smoothing_factor
        self.transforms = transforms

        self.frame = 0
        self.next_track_id = 0
        self.tracks: List[Track] = []

    def reset(self):
        self.frame = 0
        self.next_track_id = 0
        self.tracks = []

    @torch.no_grad()
    def step_batch(self, images: torch.Tensor, output_dir=None, **kwargs):
        """

        Args
            img: single image in CHW format
            kwargs: override post-processing config parameters e.g. nms_kernel, num_tracks
        """
        device = kwargs.get("device", self.device)
        transforms = kwargs.get("transforms", self.transforms)
        nms_kernel = kwargs.get("nms_kernel", self.nms_kernel)
        num_detections = kwargs.get("num_detections", self.num_detections)
        
        # forward pass
        self.model.eval()
        self.model.to(device)
        if output_dir is not None:
            images_np = images.cpu().numpy().transpose(0,2,3,1)
            images_np = np.ascontiguousarray(images_np)
        
        images = images.to(device)
        if transforms is not None:
            images = transforms(images)

        time0 = time.time()
        heatmap, box_2d, reid = self.model(images)
        model_time = time.time() - time0

        time0 = time.time()
        # gather new detections and their embeddings
        new_detections = self.model.decode_tracking(heatmap, box_2d, reid, nms_kernel=nms_kernel, num_detections=num_detections, normalize_bbox=True)
        decode_time = time.time() - time0

        time0 = time.time()
        for b in range(images.shape[0]):
            bboxes = new_detections["bboxes"][b]
            labels = new_detections["labels"][b]
            scores = new_detections["scores"][b]
            embeddings = new_detections["embeddings"][b]
            self.update(bboxes, labels, scores, embeddings, **kwargs)

            if output_dir is not None:
                bboxes = np.stack([x.bbox.cpu().numpy() for x in self.tracks if x.active], axis=0)
                track_ids = np.stack([x.track_id for x in self.tracks if x.active], axis=0)
                img = images_np[b]
                save_path = os.path.join(output_dir, f"{self.frame}.jpg")

                draw_bboxes(img, bboxes, track_ids, normalized_bbox=True, text_color=(1,1,1))
                img = (img*255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                assert cv2.imwrite(save_path, img)

            self.frame += 1
        
        matching_time = time.time() - time0

        # remove tracks
        self.tracks = [x for x in self.tracks if not x.to_delete]

        return model_time, decode_time, matching_time

    @torch.no_grad()
    def step_single(self, img: torch.Tensor, **kwargs):
        img = img.unsqueeze(0)
        return self.step_batch(img, **kwargs)

    def update(self, bboxes, labels, scores, embeddings, **kwargs):
        detection_threshold = kwargs.get("detection_threshold", self.detection_threshold)
        matching_threshold = kwargs.get("matching_threshold", self.matching_threshold)
        device = kwargs.get("device", self.device)
        smoothing_factor = kwargs.get("smoothing_factor", self.smoothing_factor)
        
        # filter by detection threshold
        mask = scores >= detection_threshold
        bboxes = bboxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        embeddings = embeddings[mask]

        if self.tracks:
            current_embeddings = torch.stack([x.embedding for x in self.tracks if not x.to_delete], dim=0)
            current_bboxes = torch.stack([x.bbox for x in self.tracks if not x.to_delete], dim=0)
        else:
            embedding_dim = embeddings.shape[-1]
            current_embeddings = torch.zeros((1,embedding_dim), device=device)
            current_bboxes = torch.zeros((1,4), device=device)
        
        # embedding cost matrix
        cost_matrix = self.matching_cost(embeddings, bboxes, current_embeddings, current_bboxes)

        # match new detections with current active tracks
        # row is new detections, column is current tracks
        # TODO: filter by label first, then do matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        assigned_det = set()
        assigned_tracks = set()
    
        # only match if cost < threshold
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < matching_threshold:
                self.tracks[col].update(bboxes[row], embeddings[row])
                self.tracks[col].active = True

                assigned_det.add(row)
                assigned_tracks.add(col)

        unmatched_detections = [x for x in range(len(bboxes)) if x not in assigned_det]
        unmatched_tracks = [x for x in range(len(self.tracks)) if x not in assigned_tracks]

        # create new tracks from unmatched detections
        for idx in unmatched_detections:
            track = Track(self.next_track_id, bboxes[idx], labels[idx], embeddings[idx], smoothing_factor=smoothing_factor)
            self.tracks.append(track)
            self.next_track_id += 1
        
        # mark unmatched tracks as inactive
        for idx in unmatched_tracks:
            self.tracks[idx].active = False

        # increment inactive age and mark for delete
        for track in self.tracks:
            track.step()

class Track:
    def __init__(self, track_id, bbox, label, embedding, smoothing_factor=0.9, max_inactive_age=30):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.embedding = embedding
        self.smoothing_factor = smoothing_factor
        self.max_inactivate_age = max_inactive_age

        self.active = True
        self.to_delete = False
        self.inactive_age = 0

    def step(self):
        # TODO: Kalman filter
        if not self.active:
            self.inactive_age += 1

            if self.inactive_age == self.max_inactivate_age:
                self.to_delete = True

    def update(self, bbox, embedding):
        # TODO: Kalman filter
        self.bbox = bbox
        self.embedding = (1-self.smoothing_factor) * self.embedding + self.smoothing_factor * embedding

    def __repr__(self):
        return f"track id: {self.track_id}, bbox: {self.bbox}, label: {self.label}, embedding: {len(self.embedding)} dim"
