import numpy as np
import cv2

def convert_cxcywh_to_x1y1x2y2(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from cxcywh format to x1y2x2y2 format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2    # x1 = x - w/2
    bboxes[...,1] -= bboxes[...,3] / 2    # y1 = y - h/2
    bboxes[...,2] += bboxes[...,0]        # x2 = w + x1
    bboxes[...,3] += bboxes[...,1]        # y2 = h + y1

    return bboxes

def draw_detections(img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray=None, inplace: bool=True, relative_scale: bool=False, color=(255,0,0)):
    """x1y1x2y2 format
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
        text = f"{labels[i]} {scores[i]:4f}"

        text_color = (0,0,0)

        cv2.rectangle(img, pt1, pt2, color, thickness=1)
        cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1)

    return img
