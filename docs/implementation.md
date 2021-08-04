# CenterNet implementation

CenterNet consists of 3 main components

- **Backbone**: any CNN classifier e.g. ResNet-50, MobileNet v2
- **Neck**: upsample the last CNN output, and may perform feature map fusion e.g. FPN
- **Output head**: final outputs for a particular task e.g. heatmap and box regression for object detection

Since CenterNet performs single-scale detection, feature map fusion is very important to achieve good performance. It is possible to extend CenterNet to multi-scale, but that will require more complicated target sampling and potentially slower inference speed.

## Components

**Backbones**

Backbone | Description
---------|------------
`ResNetBackbone` | ResNet family from `torchvision` e.g. ResNet-18/34/50
`MobileNetBackbone` | MobileNet family from `torchvision` e.g. MobileNet v2/v3-small/v3-large
`TimmBackbone` | a thin wrapper around [timm](https://github.com/rwightman/pytorch-image-models) to use Ross Wightman models

```python
from src.models import ResNetBackbone

backbone = ResNetBackbone("resnet18", pretrained=True, frozen_stages=3)
```

Pass in `return_features=True` if you need intermediate feature map outputs for feature fusion. `.forward()` will return a list of `torch.Tensor`.

**Necks**

Neck | Description
-----|------------
`SimpleNeck` | upsample the backbone output. This is used in the original CenterNet ResNet
`FPNNeck` | upsample the backbone output, and fuse with high-resolution, intermediate feature maps from backbone
`BiFPNNeck` | an upgraded version of FPN, introduced in [EfficientDet](https://arxiv.org/abs/1911.09070). CenterNet2 also uses this new backbone
`IDANeck` | iteratively fuse consecutive feature maps from backbone until there is only 1 feature map left. See [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484). This is used in the original CenterNet with DLA-34 backbone

```python
from src.models import SimpleNeck

neck = SimpleNeck([2048])
```

**Output heads** All heads have output shape (batch_size x channels x output_dim x output_dim). Number of channels depends on each output head

Output head | Description | Number of channels
------------|-------------|-------------------
`heatmap` | compulsory, class scores at each output position | num_classes
`box_2d`| bounding box regression, predicting left, top, right, bottom distance from the center location | 4
`reid` | re-identification embedding, used in FairMOT | embedding_dim (default 64)

```python
from src.models import HeatmapHead

head = HeatmapHead(64, 2)       # last channel of neck and number of classes
```

Since the model is built entirely from a config file, you can use the config file to customize model's hyperparameters. Not all hyperparameters are customizable. Check the sample config files to see what is customizable.

## The `CenterNet` class

The `CenterNet` class is a Lightning Module. Key methods:

- `__init__()`: constructor to build the network from hyperparameters. Pass in a config dictionary, or use the helper function `build_centernet()`
- `get_encoded_outputs()`: forward pass through the network, return a dictionary. Heatmap output is before sigmoid. This is used for computing loss
- `forward()`: forward pass through the network, but return a namedtuple to make the model export-friendly. Heatmap output is after sigmoid
- `compute_loss()`: pass in the encoded outputs to calculate losses for each output head and total loss
- `decode_detections()`: pass in the encoded outputs to decode to bboxes, labels, and scores predictions

## Output heads

For each head, there are 3 things to understand

- Target output: how to create target tensor
- Loss function: which loss function to use
- Decoding: how to convert head output to detections

### Heatmap

There are two 

Focal losses are used for heatmap output. All losses here are implemented to use with logit outputs (before sigmoid) to improve numerical stability.

- [x] CornerNet focal loss: first used in CornerNet. It was called Modified focal loss in the paper. Paper: https://arxiv.org/abs/1808.01244
- [x] Quality focal loss: proposed by Generalized Focal Loss paper. It generalizes Original focal loss (Retinanet). Paper: https://arxiv.org/abs/2006.04388



### Box regression

Box losses are used for bounding box regression. Only 2D is supported for now.

- [x] IoU loss
- [x] Generalized IoU loss. Paper: https://arxiv.org/abs/1902.09630
- [x] Distance IoU loss. Paper: https://arxiv.org/abs/1911.08287
- [x] Complete IoU loss. Paper: https://arxiv.org/abs/1911.08287

### Implementation notes

Unsupported features from original CenterNet:

- **Deformable convolution (DCN)**: There are implementations from Torchvision 0.8+, Detectron2, and MMCV. However, this is not export-friendly, so I do not focus on this.
- **Deep layer aggregation (DLA)**: Available from timm

Convergence speed

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

- CenterNet convergence speed is pretty slow compared to traditional detectors. But when we look at modern one-stage detectors, it's not that bad.
- As noted by other people, this is mainly because for regression heads (size and offset), only points at ground truth boxes are used for training. There are strategies proposed to use use samples during training, such as TTFNet, FCOS, and ATSS.
