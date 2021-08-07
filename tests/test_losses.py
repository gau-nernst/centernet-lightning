import torch
from torchvision.ops import box_iou, generalized_box_iou

from centernet_lightning.losses import CornerNetFocalLossWithLogits, QualityFocalLossWithLogits
from centernet_lightning.losses import IoULoss, GIoULoss, DIoULoss, CIoULoss

EPS = 1e-6

boxes1 = torch.zeros((4,10,4))
boxes1[...,:2] = torch.rand((4,10,2))
boxes1[...,2:] = boxes1[...,:2] + torch.rand(4,10,2)

boxes2 = torch.zeros((4,10,4))
boxes2[...,:2] = torch.rand((4,10,2))
boxes2[...,2:] = boxes2[...,:2] + torch.rand(4,10,2)

class TestFocalLoss:
    def test_cornernet_focal_loss(self):
        loss = CornerNetFocalLossWithLogits()
        # some points y = 1
        # all y = 1
        # all y = 0
        pass

    def test_cornernet_focal_loss_stability(self):
        loss = CornerNetFocalLossWithLogits()
        # very negative inputs
        # very small inputs
        # very positive inputs
        pass

    def test_quality_focal_loss(self):
        loss = QualityFocalLossWithLogits()
        pass

    def test_quality_focal_loss_stability(self):
        loss = QualityFocalLossWithLogits()
        # very negative inputs
        # very small inputs
        # very positive inputs
        pass

class TestIoULoss:
    def test_basic(self):
        for LossFn in (IoULoss, GIoULoss, DIoULoss, CIoULoss):
            print(LossFn)
            loss_fn = LossFn()

            # correct shape
            assert LossFn(keepdim=True)(boxes1, boxes2).shape == (4,10,1)
            assert LossFn(keepdim=False)(boxes1, boxes2).shape == (4,10)
                        
            # commutative
            loss1 = loss_fn(boxes1, boxes2)
            loss2 = loss_fn(boxes2, boxes1)
            assert (loss1 - loss2).abs().mean() < EPS
            
            # with itself
            loss = loss_fn(boxes1, boxes1)
            assert loss.abs().mean() < EPS

    def test_iou(self):
        loss_fn = IoULoss()

        # full overlap
        box1 = torch.tensor([10,20,128,256], dtype=torch.float32)
        box2 = torch.tensor([10,20,128,256], dtype=torch.float32)
        assert loss_fn(box1, box2) == 0

        # no overlap
        box1 = torch.tensor([10,20,128,256], dtype=torch.float32)
        box2 = torch.tensor([130,230,140,270], dtype=torch.float32)
        assert loss_fn(box1, box2) == 1

        # small box inside large box
        box1 = torch.tensor([10,10,11,11], dtype=torch.float32)
        box2 = torch.tensor([0,0,100,100])
        assert loss_fn(box1, box2) == 1 - 1/10000

        # test with torchvision
        loss1 = loss_fn(boxes1[0], boxes2[0]).squeeze(-1)
        loss2 = 1 - box_iou(boxes1[0], boxes2[0]).diagonal()
        assert (loss1 - loss2).abs().mean() < EPS

    def test_giou(self):
        loss_fn = GIoULoss()
  
        # full overlap
        box1 = torch.tensor([10,20,128,256], dtype=torch.float32)
        box2 = torch.tensor([10,20,128,256], dtype=torch.float32)
        assert loss_fn(box1, box2) == 0

        # no overlap
        box1 = torch.tensor([10,20,128,256], dtype=torch.float32)
        box2 = torch.tensor([130,230,140,270], dtype=torch.float32)
        assert loss_fn(box1, box2) > 1

        # enclosed box >> union box
        box1 = torch.tensor([0,0,1,100], dtype=torch.float32)
        box2 = torch.tensor([0,0,100,1], dtype=torch.float32)
        expected = 1 - (1/199 - 1 + 199/10000)
        assert (loss_fn(box1, box2) - expected).abs() < EPS
    
        # test with torchvision
        loss1 = loss_fn(boxes1[0], boxes2[0]).squeeze(-1)
        loss2 = 1 - generalized_box_iou(boxes1[0], boxes2[0]).diagonal()
        assert (loss1 - loss2).abs().mean() < EPS
