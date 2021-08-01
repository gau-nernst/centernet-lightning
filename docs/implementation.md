# CenterNet implementation

CenterNet consists of 3 main components

- Backbone: any CNN classifier e.g. ResNet-50, MobileNet v2
- Neck: upsample the last CNN output, and may perform feature map fusion e.g. FPN
- Output head: final outputs for a particular task e.g. heatmap and box regression for object detection

Since CenterNet performs single-scale detection, feature map fusion is very important to achieve good performance. It is possible to extend CenterNet to multi-scale, but that will require more complicated target sampling and potentially slower inference speed.

## Components

Backbones:

- [x] `ResNetBackbone`: from torchvision e.g. ResNet-18/34/50
- [x] `MobileNetBackbone`: from torchvision e.g. MobileNet v2/v2
- [x] `TimmBackbone`: a thin wrapper around [timm](https://github.com/rwightman/pytorch-image-models) to access Ross Wightman models

```python
from src.models import ResNetBackbone

backbone = ResNetBackbone("resnet18")
```

Necks:

- [x] `SimpleNeck`: upsample the backbone output. This is used in the original CenterNet ResNet
- [x] `FPNNeck`: upsample the backbone output, and fuse with high-resolution, intermediate feature maps from backbone
- [x] `BiFPNNeck`: an upgraded version of FPN, introduced in the EfficientDet paper. CenterNet2 also uses this new backbone
- [x] `IDANeck`: 

```python
from src.models import SimpleNeck

neck = SimpleNeck([2048])       # last channel of the backbone
```

Output heads:

- [x] `heatmap`: compulsory, class scores at each output position
- [x] `box_2d`: bounding box regression, predicting left, top, right, bottom distance from the heatmap location
- [x] `reid`: re-identification embedding, used in FairMOT

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

## Loss functions

### Focal loss

Focal losses are used for heatmap output. All losses here are implemented to use with logit outputs (before sigmoid) to improve numerical stability.

- [x] CornerNet focal loss: first used in CornerNet. It was called Modified focal loss in the paper. Paper: https://arxiv.org/abs/1808.01244
- [x] Quality focal loss: proposed by Generalized Focal Loss paper. It generalizes Original focal loss (Retinanet). Paper: https://arxiv.org/abs/2006.04388

### Box loss

Box losses are used for bounding box regression. Only 2D is supported for now.

- [x] IoU loss
- [x] Generalized IoU loss. Paper: https://arxiv.org/abs/1902.09630
- [x] Distance IoU loss. Paper: https://arxiv.org/abs/1911.08287
- [x] Complete IoU loss. Paper: https://arxiv.org/abs/1911.08287
