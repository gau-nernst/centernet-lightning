# CenterNet

Built with PyTorch Lightning, support TorchScript and ONNX support, modular design to make it simple to swap backbones and necks.

References

- [Original CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus)
- [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet)
- [TF CenterNet](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [mmdetection CenterNet](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/centernet_head.py)

## Install

Main dependencies

- pytorch, torchvision
- numpy
- opencv-python
- pytorch-lightning
- pycocotools (to read COCO dataset. Cython is required. Use [gautamchitnis](https://github.com/gautamchitnis/cocoapi) fork to support Windows)
- albumentations (for augmentations during training)

Other dependencies

- pytest (for unit testing, not required to run)
- wandb (for Weights and Biases logging, not required to run)

Environment tested: Windows 10 and Linux (Ubuntu), python=3.8, pytorch=1.8.1, torchvision=0.9.1, cudatoolkit=11.1

### Install with conda

Create new environment

```bash
conda env create -n centernet python=3.8
conda activate centernet
```

Install pytorch. Follow the official installation instruction [here](https://pytorch.org/)

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
```

In Windows, replace `-c nvidia` with `-c conda-forge`. If you don't have NVIDIA GPU or don't need GPU support, remove `cudatoolkit=11.1` and `-c nvidia`.

Install other dependencies

```bash
pip install cython, pytorch-lightning, opencv-python, numba
pip install -U albumentations --no-binary imgaug,albumentations
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI

# optional packages
pip install pytest, wandb
```

## Usage

### Model creation

Import `build_centernet` from `models` to build a CenterNet model from a YAML file. Sample config files are provided in the `configs/` directory.

```python
from src.models import build_centernet

model = build_centernet("configs/coco_resnet34.yaml")
```

You also can load a CenterNet model directly from a checkpoint thanks to PyTorch Lightning.

```python
from src.models import CenterNet

model = CenterNet.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### Inference

To run inference on a folder of images, you can directly use `CenterNet.inference()`

```python
model = ...     # create a model as above
model.eval()    # put model in evaluation mode

img_dir = "path/to/img/dir"
img_names = ["001.jpg", "002.jpg"]

detections = model.inference(img_dir, img_names, num_detections=100)

# detections = {
#   "bboxes": bounding boxes in x1y1x2y2 format, shape (num_images x num_detections x 4)
#   "labels": class labels, shape (num_images x num_detections)
#   "scores": confidence scores, shape (num_images x num_detections)
# }
```

Results are `np.ndarray`, ready for post-processing.

Internally, `CenterNet.inference()` uses the `InferenceDataset` to load the data and apply default pre-processing (resize to 512x512, normalize with ImageNet statistics). It also convert bounding boxes' coordinates to original images' dimensions.

To run inference on an image

```python
import numpy as np
import torch
import cv2

# read image from file and normalize to [0,1]
img = cv2.imread("path/to/image")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255

# optional pre-processing: resize to 512x512 and normalize with ImageNet statistics
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
img = cv2.resize(img, (512,512))
img = (img - imagenet_mean) / imagenet_std

# required pre-processing: convert from HWC to CHW format, make it a tensor and add batch dimension
img = img.transpose(2,0,1)
img = torch.from_numpy(img).unsqueeze(0)

# create a model as above and put it in evaluation mode
model = ...     
model.eval()

with torch.no_grad():
    encoded_outputs = model(img)
    detections = model.decode_detections(encoded_outputs)
```

`detections` have the same format as above, but results are `torch.Tensor`.

## Model architecture

CenterNet consists of 3 main components

- Backbone: any CNN classifier e.g. ResNet-50, MobileNet v2
- Neck: upsample the last CNN output, and may perform feature map fusion e.g. FPN
- Output head: final outputs for a particular task e.g. heatmap and box regression for object detection

Since CenterNet performs single-scale detection, feature map fusion is very important to achieve good performance. It is possible to extend CenterNet to multi-scale, but that will require more complicated target sampling and potentially slower inference speed.

Backbones:

- [x] `ResNetBackbone`: from torchvision e.g. ResNet-18/34/50
- [x] `MobileNetBackbone`: from torchvision e.g. MobileNet v2/v2
- [ ] `TimmBackbone`: a thin wrapper around [timm](https://github.com/rwightman/pytorch-image-models) to access Ross Wightman models

```python
from src.models import ResNetBackbone

backbone = ResNetBackbone("resnet18")
```

Necks:

- [x] `SimpleNeck`: upsample the backbone output. This is used in the original CenterNet ResNet
- [x] `FPNNeck`: upsample the backbone output, and fuse with high-resolution, intermediate feature maps from backbone
- [ ] `BiFPNNeck`: not implemented. This is an upgraded version of FPN, introduced in the EfficientDet paper. CenterNet2 also uses this new backbone

```python
from src.models import SimpleNeck

neck = SimpleNeck([2048])       # last channel of the backbone
```

Output heads:

- [x] `heatmap`: compulsory, class scores at each output position
- [x] `box_2d`: bounding box regression, predicting left, top, right, bottom distance from the heatmap location
- [ ] `time_displacement`: for tracking (CenterTrack)

```python
from src.models import HeatmapHead

head = HeatmapHead(64, 2)       # last channel of neck and number of classes
```

Since the model is built entirely from a config file, you can use the config file to customize model's hyperparameters. Not all hyperparameters are customizable. Check the sample config files to see what is customizable.

### The `CenterNet` class

The `CenterNet` class is a Lightning Module. Key methods:

- `__init__()`: constructor to build the network from hyperparameters. Pass in a config dictionary, or use the helper function `build_centernet()`
- `get_encoded_outputs()`: forward pass through the network, return a dictionary. Heatmap output is before sigmoid. This is used for computing loss
- `forward()`: forward pass through the network, but return a namedtuple to make the model export-friendly. Heatmap output is after sigmoid
- `compute_loss()`: pass in the encoded outputs to calculate losses for each output head and total loss
- `decode_detections()`: pass in the encoded outputs to decode to bboxes, labels, and scores predictions

## Training

It is recommended to train the model with the train script `train.py` to train with a config file.

```bash
python train.py --config "configs/coco_resnet34.yaml"
```

You can also import the `train()` function from the train script to train in your own script. You can either pass in path to your config file, or pass in a config dictionary directly.

```python
from train import train
from src.utils import load_config

# train with config file
train("config_file.yaml")

# train with config dictionary. you can modify dict values directly
config = load_config("config_file.yaml")
config["model"]["backbone"]["name"] = "resnet50"
config["trainer"]["max_epochs"] = 10
train(config)
```

The config file specifies everything required to train the model, including model construction, dataset, augmentations and training schedule.

### Custom model architecture

You can modify the backbone, neck, and output heads in their own section in the config file

### Custom dataset

Datasets in COCO and Pascal VOC formats are supported. See the Datasets section below to ensure your folder structure is correct. Change `data_dir` and `split` accordingly. For Pascal VOC, you also need to specify `name_to_label` to map class name to class label (number)

### Custom augmentations

Currently Albumentation is used to do augmentation. Any Albumentation transformations are supported. To specify a new augmentation, simply add to the list `transforms` under each dataset

### Custom trainer

This repo uses PyTorch Lightning, so we have all the PyTorch Lightning benefits. Specify any parameters you want to pass to the `trainer` in the config file to specify the training details. For a full list of option, refer to [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)

- Training epochs: Change `max_epochs`
- Multi-GPU training (not tested): Change `gpus`
- Mixed-precision training: Change `precision` to 16

### Custom optimizer and learning rate scheduler

Change `optimizer` and `lr_scheduler` under `model`. Only optimizers and schedulers from the official PyTorch is supported (in `torch.optim`). Not all schedulers will work, since they require extra information about training schedule. To use other optimizers, modify the `configure_optimizers()` method of the class `CenterNet`

### Manual training

Since `CenterNet` is a Lightning module, you can train it like any other Lightning module. Consult PyTorch Lightning documentation for more information.

```python
import pytorch_lightning as pl

model = ...     # create a model as above

trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
)

trainer.fit(model, train_dataloader, val_dataloader)
```

## Datasets

Supported dataset formats

- COCO
- Pascal VOC

There is also `InferenceDataset` class for simple inference.

### Dataset and dataloader builder

WIP

### COCO format

Download and unzip the COCO dataset. The root dataset folder should have folders `annotations` and `images` like the following:

```bash
COCO_root
├── annotations/
│   ├── instances_val2017.json
│   ├── instances_train2017.json
│   ├── ...
├── images/
│   ├── val2017/
│   ├── train2017/
│   ├── ...
```

If you have other datasets in COCO format, make sure they also have the above folder structure.

To create a COCO dataset:

```python
from src.datasets import COCODataset

dataset = COCODataset("COCO_root", "val2017")
```

### Pascal VOC format

Folder structure:

```bash
VOC_root
├── Annotations/
│   ├── 00001.xml
│   ├── 00002.xml
│   ├── ...
├── ImageSets/
│   ├── Main/
│       ├── train.txt
│       ├── val.txt
│       ├── ...
├── JPEGImages/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── ...
```

All images inside the `JPEGImages` folder must have file extension `.jpg`. All annotation files inside the `Annotations` folder must have file extension `.xml`. Dataset splits inside `ImageSets/Main` folder must have file extension `.txt`.

The names inside a dataset split file (e.g. `train.txt`) is a list of names without file extension, separated by a line break character.

```python
from src.datasets import VOCDataset

name_to_label = {
    "person": 0,
    "table": 1,
    ...
}
dataset = VOCDataset("VOC_root", "train", name_to_label=name_to_label)
```

`name_to_label` is optional, but required for training. Since annotation files only contain the string names (e.g. `person` or `car`), you need to map them to integer labels for training.

### Inference dataset

```python
from src.datasets import InferenceDataset

dataset = InferenceDataset("path/to/img/dir", img_names=["sample_img.jpg"], resize_height=512, resize_width=512)

# dataset[0] = {
#   "image_path": full path to the image
#   "image": image tensor in CHW format
#   "original_width": original image width, to convert bboxes if needed
#   "original_height": same as above
# }
```

- `img_names` is optional. if not provided, the dataset will use all JPEG images found in the folder `img_dir`.
- `resize_height` and `resize_width`: image dimensions when input to the model. Default to 512x512, the resolution 

Inference dataset does not need a custom collate function. You can create a data loader directly from an inference dataset instance.

### Custom dataset for training

You can write your own custom dataset, as long as it conforms to the format that the model expects. A batch of input data should be a dictionary with the following key-value pairs:

- `image`: images in `CHW` format. Shape `NCHW`.
- `bboxes`: bounding boxes in `(cx,cy,w,h)` format (unit: pixel). Shape `ND4`, where `D` is the number of detections in one image.
- `labels`: labels `[0,num_classes-1]`. Shape `ND`.
- `mask`: binary mask of `0` or `1`. Since each image has different number of detections, `bboxes` and `labels` are padded so that they have the same lengths within one batch. This is used in calculating loss. Shape `ND`.

If you only need inference, only key `image` is needed.

### Dataloader and collate function

WIP

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

## Notes

### Implementation

Unsupported features from original CenterNet:

- Deformable convolution (DCN). There are implementations from Torchvision 0.8+, Detectron2, and MMCV.
- Deep layer aggregation (DLA). Available from timm

There are 2 methods to render target heatmap

- CornerNet method
- TTFNet method

### Dataset

[Safety Helmet Wearing Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/)

- Some image files have extension `.JPG` instead of `.jpg`. Rename all `.JPG` to `.jpg` to avoid issues
- The Google Driver version of the dataset has the annotation `000377.xml` with label `dog`. As noted by the author [here](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/issues/18), it should be `hat` instead.

### Training

CenterNet convergence speed is very slow. The original paper trained the model for 140 epochs on COCO2017. I haven't been able to produce good results training on COCO.

As noted by other people, this is mainly because for regression heads (size and offset), only points at ground truth boxes are used for training. TTFNet comes up with a novel way to tackle this, but it is not implemented here.
