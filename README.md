# CenterNet

Special thanks

- [Original CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus)
- [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet)
- [TF CenterNet](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Install

Main dependencies

- pytorch, torchvision
- numpy
- opencv-python
- pytorch-lightning
- pycocotools (to read COCO dataset. Cython is required. Use [gautamchitnis](https://github.com/gautamchitnis/cocoapi) fork to support Windows)
- albumentations (for augmentations during training)
- numba (to speed up some calculations on CPU)

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

Import `build_centernet_from_cfg` from `model.py` to build a CenterNet model from a YAML file. Sample config files are provided in the `config/` directory.

```python
from src.models import build_centernet_from_cfg

model = build_centernet_from_cfg("configs/coco_resnet34.yaml")
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

# if you want to run inference on all images in the folder
# import os
# img_names = [x for x in os.listdir(img_dir) if x.endswith(".jpg")]

# use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

detections = model.inference(img_dir, img_names, num_detections=100)

# detections = {
#   "bboxes": bounding boxes in cxcywh format, shape (num_images x num_detections x 4)
#   "labels": class labels, shape (num_images x num_detections)
#   "scores": confidence scores, shape (num_images x num_detections)
# }
```

Results are `np.ndarray`, ready for post-processing.

Internally, `CenterNet.inference()` uses the `InferenceDataset` to load the data and apply default pre-processing (resize to 512x512, normalize with ImageNet statistics). It also convert bounding boxes' coordinates to original images' dimensions.

To run inference on an image

```python
import torch
import cv2

img = cv2.imread("path/to/image")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# pre-processing
# resize to 512x512
# normalize with imagenet statistics

model = ...     # create a model as above
model.eval()    # put model in evaluation mode

# use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

with torch.no_grad():
    # add batch dimension and transfer to GPU if available
    batch = {"image": img.unsqueeze(0).to(device)}
    
    encoded_outputs = model(batch)
    detections = model.decode_detections(encoded_outputs)
```

`detections` have the same format as above, but results are `torch.Tensor`.

## Model architecture

The `CenterNet` is a "meta" architecture: it can take in any backbone (that outputs a feature map) and output the specified output heads in parallel.

Backbones:

- [x] `SimpleBackbone`: a base CNN network with an upsample stage. It is the ResNet model specified in the original paper.
- [x] `FPNBackbone`: a base CNN network with a feature pyramid to fuse feature maps at different resolutions.
- [ ] `IDABackbone`: not implemented
- [ ] `DLABackbone`: not implemented. The author claims this is the best backbone for CenterNet/Track
- [ ] `BiFPNBackbone`: not implemented. This is an upgraded version of FPN, introduced in the EfficientDet paper. CenterNet2 also uses this new backbone

Since CenterNet performs single-scale detection, feature map fusion is very important to achieve good performance.

Output heads:

- [x] `heatmap`: compulsory, class scores at each position
- [x] `size`: width and height regression, to determine bounding box size
- [x] `offset`: center x and y offset regression, to refine center's position
- [ ] `displacement`: for CenterTrack, not implemented

Since the model is built entirely from a config file, you can use the config file to customize model's hyperparameters. Not all hyperparameters are customizable. Check the sample config files to see what is customizable.

Base CNNs:

- [x] (torchvision) ResNet family (resnet, resnext, wide resnet)
- [x] (torchvision) MobileNet family (v2, v3-large, v3-small)

### The `CenterNet` class

The `CenterNet` class is a Lightning Module. Key methods:

- `__init__()`: constructor to build the network from hyperparameters. Pass in a config dictionary, or use the helper function `build_centernet_from_cfg()` instead.
- `forward()`: forward pass behavior returns the encoded outputs from the output heads. This includes `heatmap`, `size`, and `offset` outputs. The encoded outputs are then used for computing loss or decoding to detections.
- `compute_loss()`: pass in the encoded outputs from forward pass to calculate losses for each output head and total loss.
- `decode_detections()`: pass in the encoded outputs from forward pass to decode to bboxes, labels, and scores predictions

## Training

### Using train script

To train the model, run `train.py` and specify the config file.

```bash
python train.py --config "configs/coco_resnet34.yaml"
```

The config file specifies everything required to train the model, including model construction, dataset, augmentations and training schedule.

- Customize model architecture: See above.
- Train on custom dataset: Datasets in COCO and Pascal VOC formats are supported. See the Datasets section below.
- Add custom augmentations: Only Albumentation transformations are supported. This is because Albumentation will handle transforming bounding boxes for us. To specify a new augmentation, simply add to the list `transforms` under each dataset.
- Training epochs: Change `params/max_epochs` under `trainer`
- Optimizer: Change `optimizer` under `model`. Only optimizers from the official PyTorch is supported (in `torch.optim`). To use other optimizers, modify the `configure_optimizers()` method of the class `CenterNet`
- Learning rate schedule: Change `lr_scheduler` under `model`. The schedulers are taken from the official PyTorch, but not all are supported. To use other learning rate schedulers, modify the `configure_optimizers()` method.

You can also import the `train()` function from the train script to train in your own script. You can either pass in path to your config file, or pass in a config dictionary directly.

```python
from train import train

train("config_file.yaml")
```

As `train()` also accept a config dictionary, you can load the config file as a dictionary and directly modify its values before passing to `train()`.

```python
import yaml
from train import train

# load the config file
config_file = "config_file.yaml"
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# modify config file parameters
config["model"]["backbone"]["name"] = "resnet50"
config["trainer"]["params"]["max_epochs"] = 10

# run the training
train(config)
```

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

See the Datasets section below on how to create datasets for `CenterNet`.

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

## Notes

### Implementation

Unsupported features from original CenterNet:

- Deformable convolution (DCN). There are implementations from Torchvision 0.8+, Detectron2, and MMCV.
- Deep layer aggregation (DLA)

Notable modifications:

- Focal loss: use `torch.logsigmoid()` to improve numerical stability when calculating focal loss. [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus) and [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet) clamp input tensor.

**Target heatmap** There are two versions implemented here, one from CornerNet, which original CenterNet uses, and one from TTFNet. Since CenterNet takes a lot of work from CornerNet, I believe this target heatmap is not quite appropriate, because CenterNet and CornerNet tries to predict different types of heatmap. Moreover, the CornerNet target heatmap produces very small Gaussian kernel, which further slow down convergence.

### Dataset

[Safety Helmet Wearing Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/)

- Some image files have extension `.JPG` instead of `.jpg`. Rename all `.JPG` to `.jpg` to avoid issues
- The Google Driver version of the dataset has the annotation `000377.xml` with label `dog`. As noted by the author [here](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/issues/18), it should be `hat` instead.

### Training

CenterNet convergence speed is very slow. The original paper trained the model for 140 epochs on COCO2017. I haven't been able to produce good results training on COCO.

As noted by other people, this is mainly because for regression heads (size and offset), only points at ground truth boxes are used for training. TTFNet comes up with a novel way to tackle this, but it is not implemented here.
