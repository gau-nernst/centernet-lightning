# CenterNet

CenterNet is a strong **single-stage**, **single-scale**, and **anchor-free** object detector. This implementation is built with PyTorch Lightning, supports TorchScript and ONNX export, and has modular design to make it simple customizing the components.

References

- [Original CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus)
- [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet)
- [TF CenterNet](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [mmdetection CenterNet](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/centernet_head.py)

To read more about the architecture and code structure of this implementation, see [implementation.md](docs/implementation.md)

## Install

Clone this repo and navigate to the repo directory

```bash
git clone <THIS_REPO_GIT_URL>
cd CenterNet
```

It is recommended to install PyTorch with `conda`, but `pip` should work also

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

For more detailed instructions, see [install.md](docs/install.md)

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

**Folder of images** Use `CenterNet.inference_detection()` or `CenterNet.inference_tracking()`

```python
model = ...     # create a model as above
img_dir = "path/to/img/dir"
detections = model.inference_detection(img_dir, num_detections=100)
```

`detections` is a dictionary with the following keys:

Key | Description | Shape
----|-------------|-------
`bboxes` | bounding boxes in x1y1x2y2 format | (num_images x num_detections x 4)
`labels` | class labels | (num_images x num_detections)
`scores` | confidence scores | (num_images x num_detections)

Results are `np.ndarray`, ready for post-processing.

**Single image** This is useful when you use `CenterNet` in your own applications

```python
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# read image
img = cv2.imread("path/to/image")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# apply pre-processing: resize to 512x512 and normalize with ImageNet statistics
# use torchvision.transforms should work also
transforms = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(),
    ToTensorV2()
])
img = transforms(image=img)["image"]

# create a model as above and put it in evaluation mode
model = ...     
model.eval()

# turn off gradient calculation and do forward pass
with torch.no_grad():
    encoded_outputs = model(img.unsqueeze(0))
    detections = model.decode_detections(encoded_outputs)
```

`detections` has the same format as above, but the values are `torch.Tensor`.

Note: Due to data augmentations during training, the model is robust enough to not need ImageNet normalization in inference. You can normalize input image to `[0,1]` and CenterNet should still work fine.

### Deployment

`CenterNet` is export-friendly. You can directly export a trained model to ONNX or TorchScript (only tracing) using PyTorch Lightning API

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

### Dataset

[Safety Helmet Wearing Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/)

- Some image files have extension `.JPG` instead of `.jpg`. Rename all `.JPG` to `.jpg` to avoid issues
- The Google Driver version of the dataset has the annotation `000377.xml` with label `dog`. As noted by the author [here](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/issues/18), it should be `hat` instead.

### Training

CenterNet convergence speed is very slow. The original paper trained the model for 140 epochs on COCO2017. I haven't been able to produce good results training on COCO.

As noted by other people, this is mainly because for regression heads (size and offset), only points at ground truth boxes are used for training. TTFNet comes up with a novel way to tackle this, but it is not implemented here.
