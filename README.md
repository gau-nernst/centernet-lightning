# CenterNet

CenterNet is a strong single-stage, single-scale, and anchor-free object detector. This implementation is built with PyTorch Lightning, supports TorchScript and ONNX export, and has modular design to make it simple customizing the components.

References

- [Original CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus)
- [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet)
- [TF CenterNet](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [mmdetection CenterNet](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/centernet_head.py)

## Install

You can use the following commands to install the required dependencies.

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
pip install cython pytorch-lightning opencv-python albumentations
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
pip install git+https://github.com/gau-nernst/TrackEval.git
```

For more information, see [install.md](docs/install.md)

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

### Deployment

`CenterNet` is export-friendly. You can directly export a trained model to ONNX or TorchScript using PyTorch Lightning API

```python
import torch
from src.models import CenterNet

model = CenterNet.load_from_checkpoint("path/to/checkpoint.ckpt")
model.to_onnx("model.onnx", torch.rand((1,3,512,512)))      # export to ONNX
model.to_torchscript("model.pt")                            # export to TorchScript
```

## Training CenterNet

You can train CenterNet with the provided train script `train.py` and a config file.

```bash
python train.py --config "configs/coco_resnet34.yaml"
```

See sample config files at [configs/](configs/). To customize training, see [training.md](docs/training)

## Datasets

The following dataset formats are supported:

Detection:

- [x] [COCO](https://cocodataset.org/)
- [x] [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [x] [CrowdHuman](https://www.crowdhuman.org/)

Tracking:

- [x] [MOT](https://motchallenge.net/)
- [x] [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)

To see how to use each dataset type, see [datasets.md](docs/datasets.md)

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
