import torch
from torchvision.ops import box_iou, generalized_box_iou

from src.losses import focal_loss, iou_loss

EPS = 1e-6

class TestFocalLosses:
    def test_modified_focal_loss(self):
        loss = focal_loss.ModifiedFocalLossWithLogits()
        # some points y = 1
        # all y = 1
        # all y = 0
        pass

    def test_modified_focal_loss_stability(self):
        loss = focal_loss.ModifiedFocalLossWithLogits()
        # very negative inputs
        # very small inputs
        # very positive inputs
        pass

    def test_quality_focal_loss(self):
        loss = focal_loss.QualityFocalLossWithLogits()
        pass

    def test_quality_focal_loss_stability(self):
        loss = focal_loss.QualityFocalLossWithLogits()
        # very negative inputs
        # very small inputs
        # very positive inputs
        pass

class TestIoULosses:
    def test_basic(self):
        for LossFn in (iou_loss.CenterNetIoULoss, iou_loss.CenterNetGIoULoss, iou_loss.CenterNetCIoULoss):
            loss_fn = LossFn(keepdim=False)
            # random samples
            boxes1 = torch.rand((4,10,2))
            boxes2 = torch.rand((4,10,2))
            
            assert LossFn()(boxes1, boxes2).shape == (4,10,1)       # correct shape
            assert loss_fn(boxes1, boxes2).shape == (4,10)          # correct shape
            loss1 = loss_fn(boxes1, boxes2)
            loss2 = loss_fn(boxes2, boxes1)
            assert torch.square(loss1 - loss2).mean() < EPS         # commutative
            assert loss_fn(boxes1, boxes1).square().mean() < EPS    # with itself

    def test_iou_edge_cases(self):
        loss_fn = iou_loss.CenterNetIoULoss()

        # full overlap
        boxes1 = torch.tensor([128,256], dtype=torch.float32)
        boxes2 = torch.tensor([128,256], dtype=torch.float32)
        assert loss_fn(boxes1, boxes2) == 0

        # width/height = 0
        boxes1 = torch.tensor([128,256], dtype=torch.float32)
        for i in range(2):
            boxes2 = torch.tensor([128,256], dtype=torch.float32)
            boxes2[i] = 0
            assert torch.abs(loss_fn(boxes1, boxes2) - 1) < EPS

        # very large width/height
        boxes1 = torch.tensor([10,20], dtype=torch.float32)
        for i in range(2):
            boxes2 = torch.tensor([10,20], dtype=torch.float32)
            boxes2[i] = 1e8
            assert torch.abs(loss_fn(boxes1, boxes2) - 1) < EPS

    def test_iou_with_torchvision(self):
        boxes1_wh = torch.rand((10,2))
        boxes2_wh = torch.rand((10,2))
        
        boxes1_xyxy = torch.stack([
            1 - boxes1_wh[...,0]/2, 1 - boxes1_wh[...,1]/2,
            1 + boxes1_wh[...,0]/2, 1 + boxes1_wh[...,1]/2
        ], dim=-1)
        boxes2_xyxy = torch.stack([
            1 - boxes2_wh[...,0]/2, 1 - boxes2_wh[...,1]/2,
            1 + boxes2_wh[...,0]/2, 1 + boxes2_wh[...,1]/2
        ], dim=-1)

        loss1 = iou_loss.CenterNetIoULoss(keepdim=False)(boxes1_wh, boxes2_wh)
        loss2 = 1 - box_iou(boxes1_xyxy, boxes2_xyxy).diagonal()
        assert torch.square(loss1 - loss2).mean() < EPS

    def test_giou_edge_cases(self):
        loss_fn = iou_loss.CenterNetGIoULoss()
  
        # full overlap
        boxes1 = torch.tensor([128,256], dtype=torch.float32)
        boxes2 = torch.tensor([128,256], dtype=torch.float32)
        assert loss_fn(boxes1, boxes2) == 0

        # width/height = 0
        boxes1 = torch.tensor([128,256], dtype=torch.float32)
        for i in range(2):
            boxes2 = torch.tensor([128,256], dtype=torch.float32)
            boxes2[i] = 0
            assert torch.abs(loss_fn(boxes1, boxes2) - 1) < EPS

        # very large width/height
        boxes1 = torch.tensor([10,20], dtype=torch.float32)
        for i in range(2):
            boxes2 = torch.tensor([10,20], dtype=torch.float32)
            boxes2[i] = 1e8
            assert torch.abs(loss_fn(boxes1, boxes2) - 1) < EPS

        # enclosed box >> union box
        boxes1 = torch.tensor([10,1e8], dtype=torch.float32)
        boxes2 = torch.tensor([1e8,20], dtype=torch.float32)
        assert torch.abs(loss_fn(boxes1, boxes2) - 2) < EPS
    
    def test_giou_with_torchvision(self):
        boxes1_wh = torch.rand((10,2))
        boxes2_wh = torch.rand((10,2))
        
        boxes1_xyxy = torch.stack([
            -boxes1_wh[...,0]/2 + 1, -boxes1_wh[...,1]/2 + 1,
            boxes1_wh[...,0]/2 + 1, boxes1_wh[...,1]/2 + 1
        ], dim=-1)
        boxes2_xyxy = torch.stack([
            -boxes2_wh[...,0]/2 + 1, -boxes2_wh[...,1]/2 + 1,
            boxes2_wh[...,0]/2 + 1, boxes2_wh[...,1]/2 + 1
        ], dim=-1)

        loss1 = iou_loss.CenterNetGIoULoss(keepdim=False)(boxes1_wh, boxes2_wh)
        loss2 = 1 - generalized_box_iou(boxes1_xyxy, boxes2_xyxy).diagonal()
        assert torch.square(loss1 - loss2).mean() < EPS

    def test_ciou_edge_cases(self):
        loss_fn = iou_loss.CenterNetCIoULoss()
  
        # full overlap, same ratio
        boxes1 = torch.tensor([128,256], dtype=torch.float32)
        boxes2 = torch.tensor([128,256], dtype=torch.float32)
        assert loss_fn(boxes1, boxes2) == 0

        # width = 0
        boxes1 = torch.tensor([0,128], dtype=torch.float32)
        boxes2 = torch.tensor([128,128], dtype=torch.float32)
        assert torch.abs(loss_fn(boxes1, boxes2) - 1.05) < EPS

        # # very large width/height
        # boxes1 = torch.tensor([10,1e8], dtype=torch.float32)
        # boxes2 = torch.tensor([10,20], dtype=torch.float32)
        # assert torch.abs(loss_fn(boxes1, boxes2) - 1) < EPS

        # # enclosed box >> union box
        # boxes1 = torch.tensor([10,1e8], dtype=torch.float32)
        # boxes2 = torch.tensor([1e8,20], dtype=torch.float32)
        # assert torch.abs(loss_fn(boxes1, boxes2) - 2) < EPS
