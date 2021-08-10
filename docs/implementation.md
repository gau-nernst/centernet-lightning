# CenterNet implementation

CenterNet consists of 3 main components

- **Backbone**: any CNN classifier e.g. ResNet-50, MobileNet v2
- **Neck**: upsample the last CNN output, and may perform feature map fusion e.g. FPN
- **Output head**: final outputs for a particular task e.g. heatmap and box regression for object detection

Since CenterNet performs single-scale detection, feature map fusion is very important to achieve good performance. It is possible to extend CenterNet to multi-scale, but that will require more complicated target sampling and potentially slower inference speed.

## Components

### Backbones

Backbone | Description
---------|------------
`ResNetBackbone` | ResNet family from `torchvision` e.g. ResNet-18/34/50
`MobileNetBackbone` | MobileNet family from `torchvision` e.g. MobileNet v2/v3-small/v3-large
`TimmBackbone` | a thin wrapper around [timm](https://github.com/rwightman/pytorch-image-models) to use Ross Wightman models

```python
from centernet_lightning.models import ResNetBackbone

backbone = ResNetBackbone("resnet18", pretrained=True, frozen_stages=3)
```

Constructor

Argument | Description
---------|------------
`backbone_name` | Base backbone name e.g. `resnet34`
`pretrained` (default: `True`) | Whether to load ImageNet weights
`return_features` (default: `False`) | If `True`, the backbone will return a list of intermediate feature map outputs (for feature fusion). Otherwise, only the last feature map output is returned.
`frozen_stage` (default: `0`) | How many backbone stages to freeze, including batch norm. Not implemented in `TimmBackbone`

### Necks

Neck | Description
-----|------------
`SimpleNeck` | upsample the backbone output. This is used in the original CenterNet ResNet
`FPNNeck` | upsample the backbone output, and fuse with high-resolution, intermediate feature maps from backbone
`BiFPNNeck` | an upgraded version of FPN, introduced in [EfficientDet](https://arxiv.org/abs/1911.09070). CenterNet2 also uses this new backbone
`IDANeck` | iteratively fuse consecutive feature maps from backbone until there is only 1 feature map left. See [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484). This is used in the original CenterNet with DLA-34 backbone

```python
from centernet_lightning.models import SimpleNeck, FPNNeck

neck = SimpleNeck([512])    # feature map channels from backbone
neck = FPNNeck([64, 128, 256, 512], upsample_channels=[256, 128, 64], conv_type="normal", upsample_type="nearest")
```

Note: For necks that use feature fusion (all except `SimpleNeck`), input image dimensions must be divisble by 32 (backbone stride). This is to make sure the upsampled feature maps' dimensions match their corresponding intermediate feature maps from backbone.

### Output heads

All heads have output shape (batch_size x channels x output_dim x output_dim). Number of channels depends on the specific output head

Output head | Description | Number of channels
------------|-------------|-------------------
`heatmap` | compulsory, class scores at each output position | num_classes
`box_2d`| bounding box regression, predicting left, top, right, bottom distance from the center location | 4
`reid` | re-identification embedding, used in FairMOT | embedding_dim (default: 64)

```python
from centernet_lightning.models import HeatmapHead

head = HeatmapHead(64, 2)       # last channel of neck and number of classes
```

## The `CenterNet` class

The `CenterNet` class is a Lightning Module. Key methods:

Method | Description
-------|------------
`__init__()` | Constructor to build CenterNet from hyperparameters. Pass in a config dictionary, or use the helper function `build_centernet()`.
`get_encoded_outputs()` | Forward pass and return a dictionary. The keys depend on the task. Heatmap output is before sigmoid (logits). This is used in training i.e. computing loss.
`forward()` | Forward pass and return a namedtuple to make the model export-friendly. Heatmap output is after sigmoid (confidence score). This is used in inference.
`compute_loss()` | Pass in the encoded outputs to calculate losses for each output head and total loss
`gather_detection2d/tracking2d()` | Pass in the encoded outputs to gather top-k center points from heatmap, and decode to bboxes, labels, and scores predictions (also ReID embedding for tracking).
`inference_detection2d/tracking2d()` | Run inference on a folder of images.

## Implementation notes

### Output heads

For each head, there are 3 things to understand

- Target output: how to create target tensor from detections
- Loss function: which loss function to use
- Decoding: how to convert head output to detections

#### Heatmap

There are two methods to render target heatmap: CornerNet and TTFNet. They both place a 2D Gaussian at each detection's center. They only differ in how they calculate the Gaussian size.

Heatmap is trained with focal losses. The original focal loss (from RetinaNet paper) only accepts binary values (0 or 1) for target heatmap, so people have adapted it for target with continuous values [0,1].
- CornerNet focal loss: proposed by [CornerNet paper](https://arxiv.org/abs/1808.01244), used in original CenterNet.
- Quality focal loss: proposed by [Generalized Focal Loss paper](https://arxiv.org/abs/2006.04388).

Both losses are implemented to use with logit outputs (before sigmoid) to improve numerical stability.

#### Box regression

Originally CenterNet predicts center xy offset and box width/height for box regression, which is similar to CornerNet. To align with modern one-stage detectors and simplify box output head, box regression output implemented here predicts left, top, right, bottom offsets from the center. CenterNet+, which was briefly introduced in CenterNet2 paper, also uses this box regression design.

Box losses

- L1 and Smooth L1 loss: Used in CornetNet (Smooth L1) and original CenterNet (L1)
- IoU loss
- [Generalized IoU](https://arxiv.org/abs/1902.09630) loss
- [Distance and Complete IoU](https://arxiv.org/abs/1911.08287) loss

Most modern one-stage detectors are trained with IoU-based losses. From my observations, L1 loss is more stable to train, but IoU-based losses give better final results.

Note: IoU, GIoU, DIoU and CIoU implementations here do not check for invalid boxes. Make sure boxes are valid when used in training, otherwise the training will crash.

#### Re-identification

This output head is adapted from FairMOT. Following FairMOT recommendations, the default embedding size is 64.

As this head is a re-identification problem, works related to re-identification will apply here. 

Only classification loss (cross entropy) has been tested.

### Unsupported features from original CenterNet

- **Deformable convolution (DCN)**: There are implementations from Torchvision 0.8+, Detectron2, and MMCV. However, this is not export-friendly, so I do not focus on this.
- **Deep layer aggregation (DLA)**: Available from timm

### Convergence speed

Model | Epochs on COCO2017 (official)
------|------------------------------
CenterNet | 140
Faster R-CNN | 60 epochs (on COCO2014 (?))
RetinateNet | 12
FCOS | 12
YOLOv1-v3 | 160 (not sure)
YOLOv3 | 300
YOLOX | 300
nanodet | 280

CenterNet convergence speed is pretty slow compared to traditional detectors. But when we look at modern one-stage detectors, it's not that bad.

As noted by other people, this is mainly because for regression heads (size and offset), only points at ground truth boxes are used for training. There are strategies proposed to use use samples during training, such as TTFNet, FCOS, and ATSS.
