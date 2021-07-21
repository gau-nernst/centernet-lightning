from typing import List

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

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

class Tracker:
    # Tracktor: https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
    # DeepSORT: https://github.com/ZQPei/deep_sort_pytorch/blob/master/deep_sort/sort/tracker.py

    def __init__(self, model: nn.Module, device="cpu", nms_kernel=3, num_detections=100, detection_threshold=0.1, reid_threshold=0.2):
        self.model = model
        self.device = device
        self.nms_kernel = nms_kernel
        self.num_detections = num_detections
        self.detection_threshold = detection_threshold
        self.reid_threshold = reid_threshold

        self.next_track_id = 0
        self.tracks: List[Track] = []

    @torch.no_grad()
    def step(self, img: torch.Tensor, **kwargs):
        """

        Args
            img: single image in CHW format
            kwargs: override post-processing config parameters e.g. nms_kernel, num_tracks
        """
        # forward pass
        self.model.eval()
        device = kwargs.get("device", self.device)
        self.model.to(device)
        img = img.unsqueeze(0).to(device)
        heatmap, box_2d, reid = self.model(img)

        # gather new detections and their embeddings
        nms_kernel = kwargs.get("nms_kernel", self.nms_kernel)
        num_detections = kwargs.get("num_detections", self.num_detections)
        new_detections = self.model.decode_tracking(heatmap, box_2d, reid, nms_kernel=nms_kernel, num_detections=num_detections, normalize_bbox=True)

        bboxes = new_detections["bboxes"][0]
        labels = new_detections["labels"][0]
        scores = new_detections["scores"][0]
        embeddings = new_detections["embeddings"][0]

        # filter by detection threshold
        detection_threshold = kwargs.get("detection_threshold", self.detection_threshold)
        mask = scores >= detection_threshold
        bboxes = bboxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        embeddings = embeddings[mask]

        # embedding cost matrix
        embedding_dim = embeddings.shape[-1]
        current_embeddings = torch.stack([x.embedding for x in self.tracks], dim=0) if self.tracks else torch.zeros((1,embedding_dim))
        cost_matrix = cosine_distance_matrix(embeddings, current_embeddings)

        # match new detections with current active tracks
        # row is new detections, column is current tracks
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        reid_threshold = kwargs.get("reid_threshold", self.reid_threshold)
        assigned_det = set()
        assigned_tracks = set()
        
        # only match if cost < threshold
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < reid_threshold:
                self.tracks[col].update(bboxes[row], embeddings[row])
                self.tracks[col].active = True

                assigned_det.add(row)
                assigned_tracks.add(col)

        unmatched_detections = [x for x in range(len(bboxes)) if x not in assigned_det]
        unmatched_tracks = [x for x in range(len(self.tracks)) if x not in assigned_tracks]

        # create new tracks from unmatched detections
        for idx in unmatched_detections:
            track = Track(self.next_track_id, bboxes[idx], labels[idx], embeddings[idx])
            self.tracks.append(track)
            self.next_track_id += 1
        
        # mark and delete unmatched tracks
        for idx in unmatched_tracks:
            # self.tracks[idx].to_delete = True
            self.tracks[idx].active = False
        # self.tracks = [x for x in self.tracks if not x.to_delete]

class Track:
    def __init__(self, track_id, bbox, label, embedding, smoothing_factor=0.9):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.embedding = embedding
        self.smoothing_factor = smoothing_factor
        
        self.active = True
        self.to_delete = False

    def update(self, bbox, embedding):
        self.bbox = bbox
        self.embedding = (1-self.smoothing_factor) * self.embedding + self.smoothing_factor * embedding

    def __repr__(self):
        return f"track id: {self.track_id}, bbox: {self.bbox}, label: {self.label}, embedding: {len(self.embedding)} dim"
