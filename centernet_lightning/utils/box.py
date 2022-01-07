import numpy as np
import torch


def convert_box_format(boxes, old_format, new_format):
    box_formats = ("xyxy", "xywh", "cxcywh")
    assert old_format in box_formats
    assert new_format in box_formats

    if isinstance(boxes, (torch.Tensor, np.ndarray)):
        boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else boxes.copy()
        
        # convert to xywh
        if old_format == "xyxy":
            boxes[...,2] -= boxes[...,0]        # w = x2 - x1
            boxes[...,3] -= boxes[...,1]        # h = y2 - y1
        elif old_format == "cxcywh":
            boxes[...,0] -= boxes[...,2] / 2    # x = cx - w/2
            boxes[...,1] -= boxes[...,3] / 2    # y = cy - h/2

        if new_format == "xyxy":
            boxes[...,2] += boxes[...,0]        # x2 = x1 + w
            boxes[...,3] += boxes[...,1]        # y2 = x1 + h
        elif new_format == "cxcywh":
            boxes[...,0] += boxes[...,2] / 2    # cx = x + w/2
            boxes[...,1] += boxes[...,3] / 2    # cy = y + h/2

    else:
        if isinstance(boxes[0], int):
            # convert to xywh
            if old_format == "xyxy":
                x, y, x2, y2 = boxes
                w, h = x2 - x, y2 - y
            elif old_format == "cxcywh":
                cx, cy, w, h = boxes
                x, y = cx - w/2, cy - h/2
            
            if new_format == "xyxy":
                boxes = (x, y, x+w, y+h)
            elif new_format == "cxcywh":
                boxes = (x+w/2, y+h/2, w, h)

        else:
            boxes = (convert_box_format(x, old_format, new_format) for x in boxes)

    return boxes


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
