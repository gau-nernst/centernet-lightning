import numpy as np

def convert_xywh_to_cxcywh(bboxes: np.ndarray, inplace=False):
    """Convert bboxes from xywh format to cxcywh format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()

    bboxes[...,0] += bboxes[...,2] / 2      # cx = x + w/2
    bboxes[...,1] += bboxes[...,3] / 2      # cy = y + h/2
    return bboxes

def convert_cxcywh_to_xywh(bboxes: np.ndarray, inplace=False):
    """Convert bboxes from cxcywh format to xywh format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2      # x = cx - w/2
    bboxes[...,1] -= bboxes[...,3] / 2      # y = cy - h/2
    return bboxes

def convert_xywh_to_x1y1x2y2(bboxes: np.ndarray, inplace=False):
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,2] += bboxes[...,0]          # x2 = x1 + w
    bboxes[...,3] += bboxes[...,1]          # y2 = x1 + h
    return bboxes

def convert_x1y1x2y2_to_xywh(bboxes: np.ndarray, inplace=False):
    if not inplace:
        bboxes = bboxes.copy()

    bboxes[...,2] -= bboxes[...,0]          # w = x2 - x1
    bboxes[...,3] -= bboxes[...,1]          # h = y2 - y1
    return bboxes

def convert_cxcywh_to_x1y1x2y2(bboxes: np.ndarray, inplace=False):
    """Convert bboxes from cxcywh format to x1y2x2y2 format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    convert_cxcywh_to_xywh(bboxes, inplace=True)
    convert_xywh_to_x1y1x2y2(bboxes, inplace=True)
    return bboxes

def convert_x1y1x2y2_to_cxcywh(bboxes: np.ndarray, inplace=False):
    if not inplace:
        bboxes = bboxes.copy()
    
    convert_x1y1x2y2_to_xywh(bboxes, inplace=True)
    convert_xywh_to_cxcywh(bboxes, inplace=True)
    return bboxes
