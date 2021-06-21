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
- albumentations (for augmentations)

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
pip install cython, pytorch-lightning, opencv-python
pip install -U albumentations --no-binary imgaug,albumentations
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI

# optional packages
pip install pytest, wandb
```

## Usage

Import `build_centernet_from_cfg` from `model.py` to create a CenterNet model from a YAML file. Sample config files are provided in the `config/` directory.

```python
from model import build_centernet_from_cfg

model = build_centernet_from_cfg("configs/coco_resnet34.yaml")
```

To run inference

```python
import cv2

img = cv2.imread("sample_img.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (1,2,0))        # HWC to CHW
img = {"image": torch.from_numpy(img).unsqueeze(1)}

model.eval()
with torch.no_grad():
    encoded_outputs = model(img)
    detections = model.decode_detections(encoded_outputs, num_detections=10)

# detections = {
#   "bboxes": bounding boxes in cxcywh format, shape (batch x num_detections x 4)
#   "labels": class labels, shape (batch x num_detections)
#   "scores": confidence scores, shape (batch x num_detections)
# }
```

Results are `torch.Tensor`. Use `.numpy()` to get `np.ndarray`.

## Model architecture

The `CenterNet` is a "meta" architecture: it can take in any backbone (that outputs a feature map) and output the specified output heads in parallel. The `CenterNet` class also implements a few functions to help with training and running detections.

Backbones:

- [x] `SimpleBackbone`: a base CNN network with an upsample stage. It is the ResNet model specified in the original paper.
- [x] `FPNBackbone`: a base CNN network with a feature pyramid to fuse feature maps at different resolutions.
- [ ] `IDABackbone`: not implemented
- [ ] `DLABackbone`: not implemented. The author claims this is the best backbone for CenterNet/Track
- [ ] `BiFPNBackbone`: not implemented. This is an upgraded version of FPN, introduced by the EfficientDet paper. CenterNet2 also uses this new backbone

Since CenterNet performs single-scale detection, feature map fusion is very important to achieve good performance.

Output heads:

- [x] `heatmap`: compulsory, class scores at each position
- [x] `size`: (relative) width and height regression, to determin bounding box size
- [x] `offset`: center x and y offset regression, to refine center's position
- [ ] `displacement`: for CenterTrack, not implemented

Since the model is built entirely from a config file, you can use the config file to customize model's hyperparameters. Not all hyperparameters are customizable. Check the sample config files to see what is customizable.

Base CNNs:

- [x] (torchvision) ResNet family (resnet, resnext, wide resnet)
- [x] (torchvision) MobileNet family (v2, v3-large, v3-small)

## Training

To train the model, run `train.py` and specify the config file.

```bash
python train.py --config "configs/coco_resnet34.yaml"
```

The config file specifies everything required to train the model, including model construction, dataset, augmentations and training schedule.

- Customize model architecture: See above.
- Train on custom dataset: Not possible to specify in the config file for now. See below on how to write your custom dataset.
- Add custom augmentations: Only Albumentation transformations are supported. This is because Albumentation will handle transforming bounding boxes for us. To specify a new augmentation, simply add to the list `transforms` under each dataset.
- Training epochs: You can change it in the config file
- Optimizer: Optimizers are taken from the official PyTorch library, so you can specify any of them in the config file. To use other optimizers, modify the `configure_optimizers()` method of the class `CenterNet`
- Learning rate schedule: Similar to optimizers, learning rate schedulers are takken from the official PyTorch library. To use other learning rate schedulers, modify the `configure_optimizers()` method.

## Dataset

Currently only COCO dataset is supported for training. A simple `InferenceDataset` class can be used for inference only.

### COCO dataset

Download and unzip the COCO dataset. It should have the following folder structure

```bash
COCO
├── annotations/
│   ├── instances_val2017.json
│   ├── instances_train2017.json
│   ├── ...
├── val2017/
├── train2017/
```

Lightning Datamodule is used to prepare the dataset. Pass in a config dictionary to initialize the data module. Check sample config files for the structure and possible options.

```python
from datasets import COCODataModule

train_cfg = {
    "data_dir": "datasets/COCO",
    "coco_version": "train2017",
    "dataloader_params": {"batch_size": 4}
}
val_cfg = {
    "data_dir": "datasets/COCO",
    "coco_version": "val2017",
    "dataloader_params": {"batch_size": 4}
}

coco_datamodule = COCODataModule(train=train_cfg, validation=val_cfg)
coco_datamodule.setup()         # process coco annotations

train_dataloader = coco_datamodule.train_dataloader()
val_dataloader = coco_datamodule.val_dataloader()
```

`pycocotools` loads all annotation data into memory, which can be very large for COCO train set. `.setup()` method will run `prepare_coco_detection()` function to extract only the necessary information for the detection task (image file paths, bounding boxes and labels). 

You can also directly use the class `COCODataset`, but make sure to run `prepare_coco_detection()` before creating the dataset.

```python
from datasets import prepare_coco_detection, COCODataset

coco_dir = "datsets/COCO"
prepare_coco_detection(coco_dir, "val2017")
val_coco_ds = COCODataset(coco_dir, "val2017", transforms=...)
```

### Inference dataset

```python
from datasets import InferenceDataset

dataset = InferenceDataset("path/to/images")
```

### Custom dataset for training

You can write your own custom dataset, as long as it conforms to the format that the model expects. A batch of input data should be a dictionary with the following key-value pairs:

- `image`: images in `CHW` format. Shape `NCHW`.
- `bboxes`: bounding boxes in `(cx,cy,w,h)` format (unit: pixel). Shape `ND4`, where `D` is the number of detections in one image.
- `labels`: labels `[0,num_classes-1]`. Shape `ND`.
- `mask`: binary mask of `0` or `1`. Since each image has different number of detections, `bboxes` and `labels` are padded so that they have the same lengths within one batch. This is used in calculating loss. Shape `ND`.

If you only need inference, only key `image` is needed.

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
