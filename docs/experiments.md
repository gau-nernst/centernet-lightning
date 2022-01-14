# Experiments

## January 2022

Config

- Dataset: COCO 2017, 512x512
- Train augmentations: HorizontalFlip, RandomResizedCrop, ColorJitter (0.4)
- Optimizer: AdamW
- Learning rate: 5e-4 for batch size 128. Use linear scaling for other batch sizes (`lr = 5e-4*batch_size/128`)
- Weight decay: 1e-3
- Epochs: 100
- LR schedule: linear warmup (5 epochs), then cosine annealing
- GPU: 2x 3090 DDP
- SyncBN

Notes:

- AP small / medium / large are reported on 512 x 512 size, not the original sizes.
- ^ means "same as above"

Backbone | Neck | Head | Heatmap | Box | Batch size | mAP | AP small | AP medium | AP large
---------|------|------|---------|-----|------------|-----|----------|-----------|----------
ResNet-34 (21.3M) | FPN (dim=128, 0.6M) | w=128, d=2 (0.6M) | radius=cornernet | multiplier=16, loss=L1, loss_weight=0.1 | 128 | 18.6 | 30.2 | 14.9 | 3.4
VoVNet-39 (25.2M) | FPN (dim=256, 2.4M) | w=256, d=3 (3.6M) | ^ | multiplier=16, loss=GIoU, loss_weight=5. | 64 | 34.6 | 50.9 | 32.8 | 9.5
^ | ^ | ^ | ^ | ^ and 3x3 center sampling for box regression | ^ | 37.3 | 52.4 | 35.0 | 13.4
^ | ^ | ^ | radius=2 | ^ | ^ | xx.x | xx.x | xx.x | xx.x

## August 2021

### Experiment 1

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

### Experiment 2: Deformable convolution

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

### Experiment 3: LR scheduler

- Neck: FPN
- Learning rate: 5e-4
- Other config same as Experiment 1

Optimizer | LR scheduler | mAP | Remarks
----------|--------------|-----|---------
AdamW | OneCycleLR | 25.8 | Baseline
AdamW | MultiStepLR | 24.9 | -0.1 mAP
AdamW | None | 24.3 | -0.6 mAP

### Experiment 4: Mobile models

- Neck: FPN (with depthwise separable convolution)
- Batch size: 64
- Learning rate: 2.5e-4
- Other config same as Experiment 1

Backbone | mAP
---------|-----
MobileNetv2 | 15.4
MobileNetv3-large | 18.9
