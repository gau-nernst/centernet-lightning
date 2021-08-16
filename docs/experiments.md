# Experiments

**Experiment 1**

- Dataset: Pascal VOC 2012, random 80-20 split from `trainval` set
- Backbone: ResNet-18, pre-trained on ImageNet
- Batch size: 128
- Weight decay: 1e-4
- LR scheduler: OneCycleLR
- GPU: 3090 (24GB VRAM)

Neck | Optimizer | LR | mAP | Remarks
-----|-----------|----|-----|---------
FPN | Adam | 5e-4 | null | Loss explodes at epoch 79
FPN | SGD | 5e-2 | 1.1 | 
FPN | AdamW | 5e-4 | 25.8 | Baseline
FPN | AdamW | 5e-3 | 24.3 | -1.5 mAP
Weighted FPN | AdamW | 5e-4 | 26.3 | +0.5 mAP 
FPN with DCNv2 | AdamW | 5e-4 | 37.5 | +11.7 mAP
Bi-FPN | AdamW | 2.5e-4 | 32.1 | +6.3 mAP, Half batch size
IDA | AdamW | 5e-4 | 27.6 | +1.8 mAP
Bi-FPN with DCNv2 | AdamW | 2.5e-4 | 39.4 | +13.6 mAP, Half batch size

**Experiment 2** Deformable convolution

- Neck: FPN
- Optimizer: AdamW
- Learning rate: 5e-4
- Other config same as Experiment 1

DCN version | Mask activation | mAP | Remarks
------------|-----------------|-----|--------
DCNv2 | sigmoid | 37.5 | Baseline
DCNv1 | sigmoid | 37.5 | +0.0 mAP
DCNv2 | relu | 36.7 | -0.8 mAP
DCNv2 | hard sigmoid | 37.3 | -0.2 mAP

**Experiment 3** LR scheduler

- Neck: FPN
- Learning rate: 5e-4
- Other config same as Experiment 1

Optimizer | LR scheduler | mAP | Remarks
----------|--------------|-----|---------
AdamW | OneCycleLR | 25.8 | Baseline
AdamW | MultiStepLR | 24.9 | -0.1 mAP
AdamW | None | 24.3 | -0.6 mAP

**Experiment 4** Mobile models

- Neck: FPN (with depthwise separable convolution)
- Batch size: 64
- Learning rate: 2.5e-4
- Other config same as Experiment 1

Backbone | mAP
---------|-----
MobileNetv2 | 15.4
MobileNetv3-large | 18.9
