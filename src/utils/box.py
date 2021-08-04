from typing import Union

import numpy as np
import torch

def convert_xywh_to_cxcywh(bboxes: Union[np.ndarray, torch.Tensor], inplace=False):
    """Convert bboxes from xywh format to cxcywh format
    """
    if not inplace:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone()
        else:
            bboxes = bboxes.copy()

    bboxes[...,0] += bboxes[...,2] / 2      # cx = x + w/2
    bboxes[...,1] += bboxes[...,3] / 2      # cy = y + h/2
    return bboxes

def convert_cxcywh_to_xywh(bboxes: Union[np.ndarray, torch.Tensor], inplace=False):
    """Convert bboxes from cxcywh format to xywh format
    """
    if not inplace:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone()
        else:
            bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2      # x = cx - w/2
    bboxes[...,1] -= bboxes[...,3] / 2      # y = cy - h/2
    return bboxes

def convert_xywh_to_x1y1x2y2(bboxes: Union[np.ndarray, torch.Tensor], inplace=False):
    """Convert bboxes from xywh format to x1y2x2y2 format
    """
    if not inplace:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone()
        else:
            bboxes = bboxes.copy()
    
    bboxes[...,2] += bboxes[...,0]          # x2 = x1 + w
    bboxes[...,3] += bboxes[...,1]          # y2 = x1 + h
    return bboxes

def convert_x1y1x2y2_to_xywh(bboxes: Union[np.ndarray, torch.Tensor], inplace=False):
    """Convert bboxes from x1y2x2y2 format to xywh format
    """
    if not inplace:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone()
        else:
            bboxes = bboxes.copy()

    bboxes[...,2] -= bboxes[...,0]          # w = x2 - x1
    bboxes[...,3] -= bboxes[...,1]          # h = y2 - y1
    return bboxes

def convert_cxcywh_to_x1y1x2y2(bboxes: Union[np.ndarray, torch.Tensor], inplace=False):
    """Convert bboxes from cxcywh format to x1y2x2y2 format
    """
    if not inplace:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone()
        else:
            bboxes = bboxes.copy()
    
    convert_cxcywh_to_xywh(bboxes, inplace=True)
    convert_xywh_to_x1y1x2y2(bboxes, inplace=True)
    return bboxes

def convert_x1y1x2y2_to_cxcywh(bboxes: Union[np.ndarray, torch.Tensor], inplace=False):
    """Convert bboxes from x1y2x2y2 format to cxcywh format
    """
    if not inplace:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone()
        else:
            bboxes = bboxes.copy()
    
    convert_x1y1x2y2_to_xywh(bboxes, inplace=True)
    convert_xywh_to_cxcywh(bboxes, inplace=True)
    return bboxes

def box_inter_union_matrix(boxes1: np.ndarray, boxes2: np.ndarray):
    area1 = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    area2 = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)          # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]   # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union

def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray):
    inter, union = box_inter_union_matrix(boxes1, boxes2)
    iou = inter / union
    return iou

def box_giou_matrix(boxes1: np.ndarray, boxes2: np.ndarray):
    inter, union = box_inter_union_matrix(boxes1, boxes2)
    iou = inter / union

    lti = np.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rbi = np.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    whi = (rbi - lti).clip(min=0)       # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai

def box_iou_distance_matrix(boxes1: np.ndarray, boxes2: np.ndarray):
    """1 - IoU
    """
    return 1 - box_iou_matrix(boxes1, boxes2)

def box_giou_distance_matrix(boxes1: np.ndarray, boxes2: np.ndarray):
    """1 - GIoU
    """
    return 1 - box_giou_matrix(boxes1, boxes2)
