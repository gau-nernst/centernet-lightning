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
