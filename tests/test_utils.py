import numpy as np
import torch
from src.utils import *

box_convert_fn = [
    convert_xywh_to_x1y1x2y2,
    convert_x1y1x2y2_to_xywh,
    convert_xywh_to_cxcywh,
    convert_cxcywh_to_xywh,
    convert_x1y1x2y2_to_cxcywh,
    convert_cxcywh_to_x1y1x2y2
]

class TestBoxConversion:
    xywh_box     = np.array([10,20,30,40], dtype=np.float32)
    x1y1x2y2_box = np.array([10,20,40,60], dtype=np.float32)
    cxcywh_box   = np.array([25,40,30,40], dtype=np.float32)

    def test_inplace(self):
        box1 = np.random.rand(4,10,4)
        for convert_fn in box_convert_fn:
            box2 = convert_fn(box1)
            assert box2 is not box1
            box2 = convert_fn(box1, inplace=True)
            assert box2 is box1

    def test_xywh_and_x1y1x2y2(self):
        new_box = convert_xywh_to_x1y1x2y2(self.xywh_box, inplace=False)
        assert np.all(new_box == self.x1y1x2y2_box)
        new_box = convert_x1y1x2y2_to_xywh(self.x1y1x2y2_box, inplace=False)
        assert np.all(new_box == self.xywh_box)

    def test_xywh_and_cxcywh(self):
        new_box = convert_xywh_to_cxcywh(self.xywh_box, inplace=False)
        assert np.all(new_box == self.cxcywh_box)
        new_box = convert_cxcywh_to_xywh(self.cxcywh_box, inplace=False)
        assert np.all(new_box == self.xywh_box)
    
    def test_x1y1x2y2_and_cxcywh(self):
        new_box = convert_x1y1x2y2_to_cxcywh(self.x1y1x2y2_box, inplace=False)
        assert np.all(new_box == self.cxcywh_box)
        new_box = convert_cxcywh_to_x1y1x2y2(self.cxcywh_box, inplace=False)
        assert np.all(new_box == self.x1y1x2y2_box)

def test_draw_bboxes():
    img = np.random.rand(512,512,3)
    bboxes = np.random.rand(16,4)
    labels = np.random.randint(0, 79, size=(16,2))
    scores = np.random.rand(16)
    
    img_new = draw_bboxes(img, bboxes, labels, inplace=False)
    assert img_new is not img
    img_test = img.copy()
    img_new = draw_bboxes(img_test, bboxes, labels)
    assert img_new is img_test

    draw_bboxes(img, bboxes, labels)
    draw_bboxes(img, bboxes, labels, scores)
    draw_bboxes(img, bboxes, labels, scores, score_threshold=0.5)
    draw_bboxes(img, bboxes, labels, normalized_bbox=True)
    draw_bboxes(img, bboxes, labels, color=(0,1,1))
    draw_bboxes(img, bboxes, labels, text_color=(1,1,1))

def test_apply_mpl_cmap():
    img = np.random.rand(4,128,128)
    img_new = apply_mpl_cmap(img, "viridis")
    assert isinstance(img_new, np.ndarray)
    assert img_new.shape == (4,128,128,3)

    img_new = apply_mpl_cmap(img, "viridis", return_tensor=True)
    assert isinstance(img_new, torch.Tensor)
    
    img_new = apply_mpl_cmap(img, "viridis", channel_first=True)
    assert img_new.shape == (4,3,128,128)

# def test_log_image_callback():
#     config = None
#     callback = LogImageCallback(config)
#     callback = LogImageCallback(config, 10)
#     callback = LogImageCallback(config, range(10))

def test_make_image_grid():
    num_samples = 16
    imgs = [np.random.rand(128,128,3) for _ in range(num_samples)]
    bboxes1 = []
    bboxes2 = []
    for _ in range(num_samples):
        num_dets = np.random.randint(0,10)
        bboxes1.append(np.random.rand(num_dets,4))
        num_dets = np.random.randint(0,10)
        bboxes2.append(np.random.rand(num_dets,4))

    make_image_grid(imgs)
    make_image_grid(imgs, bboxes1)
    make_image_grid(imgs, bboxes1, bboxes2)

def test_convert_bboxes_to_wandb():
    num_boxes = 16
    bboxes = np.random.rand(num_boxes, 4)
    labels = np.random.rand(num_boxes)
    scores = np.random.rand(num_boxes)

    convert_bboxes_to_wandb(bboxes, labels)
    convert_bboxes_to_wandb(bboxes, labels, scores)
