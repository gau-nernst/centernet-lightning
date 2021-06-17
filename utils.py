import numpy as np
import cv2

def convert_cxcywh_to_x1y1x2y2(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from cxcywh format to x1y2x2y2 format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2    # x1 = cx - w/2
    bboxes[...,1] -= bboxes[...,3] / 2    # y1 = cy - h/2
    bboxes[...,2] += bboxes[...,0]        # x2 = w + x1
    bboxes[...,3] += bboxes[...,1]        # y2 = h + y1

    return bboxes

def convert_cxcywh_to_xywh(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from cxcywh format to xywh format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2      # x = cx - w/2
    bboxes[...,1] -= bboxes[...,3] / 2      # y = cy - h/2

    return bboxes

def convert_xywh_to_cxcywh(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from xywh format to cxcywh format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()

    bboxes[...,0] += bboxes[...,2] / 2      # cx = x + w/2
    bboxes[...,1] += bboxes[...,3] / 2      # cy = y + h/2

    return bboxes

def draw_bboxes(img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray=None, inplace: bool=True, relative_scale: bool=False, color=(255,0,0)):
    """Draw bounding boxes on an image using `cv2`
    
    Args:
        `img`: RGB image in HWC format. dtype is uint8 [0,255]
        `bboxes`: x1y1x2y2 format
    """
    if not inplace:
        img = img.copy()

    if relative_scale:
        bboxes = bboxes.copy()
        bboxes[:,[0,2]] *= img.shape[1]
        bboxes[:,[1,3]] *= img.shape[0]

    if type(scores) != np.ndarray:
        scores = np.ones_like(labels)

    for i in range(bboxes.shape[0]):
        pt1 = bboxes[i,:2].astype(int)
        pt2 = bboxes[i,2:].astype(int)
        text = f"{labels[i]} {scores[i]:.4f}"

        text_color = (0,0,0)
        cv2.rectangle(img, pt1, pt2, color, thickness=1)
        cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1)

    return img

def draw_heatmap(img: np.ndarray, heatmap: np.ndarray, inplace: bool=True):
    """Draw heatmap on image. Both `img` and `heatmap` are in HWC format
    """
    if not inplace:
        img = img.copy()

    if heatmap.shape[-1] > 1:
        heatmap = np.max(heatmap, axis=-1)   # reduce to 1 channel

    # blend to first channel, using max
    img[:,:,0] = np.maximum(img[:,:,0], heatmap, out=img[:,:,0])
    return img