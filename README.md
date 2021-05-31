# CenterNet

Special thanks

- [Original CenterNet](https://github.com/xingyizhou/CenterNet)
- [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus)
- [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet)
- [TF CenterNet](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Install

Main libraries
pytorch, torchvision, pycocotools (read COCO dataset), pytorch-lightning, albumentations (for augmentations)

pytest (for testing, not required to run)

## Usage


## Training

Use `train.py`. Specify hyperparameters in config file. PyTorch Lightning is used for training.

## Dataset

You can write your own custom dataset. Currently the model takes in a dictionary with the following key-value pairs:

- `image`: images in `CHW` format. Shape `NCHW`.
- `bboxes`: bounding boxes `(x_center, y_center, width, height)` in relative sizes `[0,1]`. Shape `ND4`, where `D` is the number of detections in one image.
- `labels`: labels `[0,num_classes-1]`. Shape `ND`.
- `mask`: binary mask of `0` or `1`. Since each image has different number of detections, `bboxes` and `labels` are padded so that they have the same sizes within one batch. This is used in calculating loss. Shape `ND`.

This format is used in `forward()`, and `compute_loss()` methods

## Notes

Unsupported features from original CenterNet:

- Deformable convolution (DCN). There are implementations from Torchvision 0.8+, Detectron2, and MMCV.
- Deep layer aggregation (DLA)

Notable modifications:

- Focal loss: use `torch.logsigmoid()` to improve numerical stability when calculating focal loss. [CenterNet-better-plus](https://github.com/lbin/CenterNet-better-plus) and [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet) clamp input tensor.
- Gaussian render kernel: following [Simple-CenterNet](https://github.com/developer0hye/Simple-CenterNet), the Gaussian kernel to render ground truth heatmap follows [TTFNet](https://arxiv.org/abs/1909.00700) formulation, which is simpler (and supposedly better) than the original CenterNet (also taken from RetinaNet)
