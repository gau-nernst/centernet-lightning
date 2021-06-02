import numpy as np
import cv2

def draw_detections(img: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, inplace: bool=True, relative_scale: bool=False):
    """x1y1x2y2 format
    """
    if not inplace:
        img = img.copy()

    if relative_scale:
        bboxes = bboxes.copy()
        bboxes[:,[0,2]] *= img.shape[1]
        bboxes[:,[1,3]] *= img.shape[0]

    for i in range(bboxes.shape[0]):
        pt1 = bboxes[i,:2].astype(int)
        pt2 = bboxes[i,2:].astype(int)
        text = f"{labels[i]} {scores[i]}"

        color = (255,0,0)
        text_color = (0,0,0)

        cv2.rectangle(img, pt1, pt2, color, thickness=1)
        cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1)

    return img