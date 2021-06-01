# CenterNet

Special thanks

- [Original CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus)
- [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet)
- [TF CenterNet](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Install

Main dependencies

- pytorch, torchvision
- opencv-python
- pytorch-lightning
- pycocotools (to read COCO dataset. Cython is required. Use [gautamchitnis](https://github.com/gautamchitnis/cocoapi) fork to support Windows)
- albumentations (for augmentations during training)

Other dependencies

- pytest (for unit testing, not required to run)

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
conda install cython
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
pip install opencv-python
pip install -U albumentations --no-binary imgaug,albumentations
```

## Usage

The `CenterNet` is a "meta" architecture: it can take in any backbone (that outputs a feature map) and output the given output heads in parallel. `CenterNet` also implements a few functions to help with training and running detections.

Backbones:

- [x] `ResNetBackbone`: implemented using torchvision's ResNet. It's ResNet with an upsample stage, as specified in the original paper. DCN layer is not implemented.
- [ ] `ResNetFPNBackbone`: not implemented. The author does not implement this, but TensorFlow implements this
- [ ] `DLABackbone`: not implemented. The author claims this is the best backbone for CenterNet/Track

Output heads:

- [x] `heatmap`: compulsory, class scores at each position
- [x] `size`: width and height regression, to determin bounding box size
- [x] `offset`: center x and y offset regression, to refine center's position
- [ ] `displacement`: for CenterTrack, currently not implemented

## Training

Use `train.py`. Specify hyperparameters in config file. PyTorch Lightning is used for training.

## Dataset

Currently only COCO dataset is supported.

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

pycocotools loads all annotation data into memory, which can be very large for COCO train set, thus not efficient for pytorch DataLoader. Run `prepare_coco_detection()` function from `datasets.py` to extract only the necessary information (image file paths, bounding boxes and labels) and serialize it to disk with pickle (note: pickle is platform dependent). `COCODataset` reads this pickle file instead of the original annotation file.

```python
from datasets import prepare_coco_detection

prepare_coco_detection(coco_dir, "val2017")
prepare_coco_detection(coco_dir, "train2017)
```

COCO dataset then can be initialized

```python
from datasets import COCODataset

val_coco_ds = COCODataset(coco_dir, "val2017")
train_coco_ds = COCODataset(coco_dir, "train2017")
```

### Custom dataset

You can write your own custom dataset, as long as it conforms to the format that the model expects. The model takes in a dictionary with the following key-value pairs:

- `image`: images in `CHW` format. Shape `NCHW`.
- `bboxes`: bounding boxes `(x_center, y_center, width, height)` in relative scale `[0,1]`. Shape `ND4`, where `D` is the number of detections in one image.
- `labels`: labels `[0,num_classes-1]`. Shape `ND`.
- `mask`: binary mask of `0` or `1`. Since each image has different number of detections, `bboxes` and `labels` are padded so that they have the same sizes within one batch. This is used in calculating loss. Shape `ND`.

This format is used in `forward()`, and `compute_loss()` methods

## Notes

Unsupported features from original CenterNet:

- Deformable convolution (DCN). There are implementations from Torchvision 0.8+, Detectron2, and MMCV.
- Deep layer aggregation (DLA)

Notable modifications:

- Focal loss: use `torch.logsigmoid()` to improve numerical stability when calculating focal loss. [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus) and [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet) clamp input tensor.
- Gaussian render kernel: following [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet), the Gaussian kernel to render ground truth heatmap follows [TTFNet](https://arxiv.org/abs/1909.00700) formulation, which is simpler (and supposedly better) than the original CenterNet (which was taken from RetinaNet)
