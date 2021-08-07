# Datasets

The following dataset formats are supported:

Detection:

Format | Class name
-------|-----------
[COCO](https://cocodataset.org/) | `COCODataset`
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | `VOCDataset`
[CrowdHuman](https://www.crowdhuman.org/) | `CrowdHumanDataset`

Tracking:

Format | Class name
-------|-----------
[MOT](https://motchallenge.net/) | `MOTTrackingSequence` and `MOTTrackingDataset`
[KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) | `KITTITrackingSequence` and `KITTITrackingDataset`

## Usage

Typical usage

```python
from torch.utils.data import DataLoader
from centernet_lightning.datasets import COCODataset, CollateDetection, get_default_detection_transforms

transforms = get_default_detection_transforms()
dataset = COCODataset("datasets/COCO", "train2017", transforms=transforms)
collate_fn = CollateDetection()
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

### Constructor

Datasets use a common constructor signature

Argument | Description | Notes
---------|-------------|------
`data_dir` | Path to data directory |
`split` | Data split e.g. `train`, `val` |
`transforms` (default: `None`) | An `albumentations` transform | When `transforms=None`, dataset will return original NumPy image array from OpenCV (RGB)
`name_to_label` (default: `None`) | A dictionary that maps from string name to integer label e.g. `{"person": 0}` | Only available for Pascal VOC and KITTI Tracking. This is required for training

### Dataset item

Dataset items are dictionaries with the following keys

Key | Description | Shape
----|-------------|------
`image` | RGB images in `CHW` format | 3 x img_height x img_width
`bboxes`| Bounding boxes in `(cx,cy,w,h)` format | 4 x num_detections
`labels` | Labels `[0,num_classes-1]` | num_detections
`ids` | (only for tracking) Unique object id | num_detections
`mask` | (only in batch) Binary mask, either `0` or `1` | num_detections

### Collate function

Since images within a batch have different number of detections, we need to pad `bboxes`, `labels`, and `ids` (only for tracking) to the same size. Use `CollateDetection()` and `CollateTracking()` to pad to largest number of detections. 

Pass this to `collate_fn` argument of PyTorch `DataLoader`. Each batch will have an extra key `mask`, which is a binary mask for each image's set of detections.

## Detection datasets

### COCO

Folder structure:

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

To create a COCO dataset:

```python
from centernet_lightning.datasets import COCODataset

dataset = COCODataset("COCO_root", "val2017")
```

### Pascal VOC

Folder structure:

```bash
VOC_root
├── Annotations/
│   ├── 00001.xml
│   ├── 00002.xml
│   ├── ...
├── ImageSets/
│   ├── Main/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── ...
├── JPEGImages/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── ...
```

Make sure file extensions are correct.

The dataset split (e.g. `train.txt`) contains a list of image names without file extension. Each name is on a separate line.

To create a Pascal VOC dataset:

```python
from centernet_lightning.datasets import VOCDataset

name_to_label = {
    "person": 0,
    "table": 1,
    ...
}
dataset = VOCDataset("VOC_root", "train", name_to_label=name_to_label)
```

`name_to_label` is optional, but required for training. Since annotation files only contain the string names (e.g. `person` or `car`), you need to map them to integer labels for training.

### CrowdHuman

Folder structure:

```bash
CrowdHuman_root
├── train/
│   ├── annotation_train.odgt
│   ├── Images/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── ...
├── val/
│   ├── ...
```

By default, class `mask` is ignored. To include class `mask`, pass in `ignore_mask=False`

## Tracking datasets

### MOT

Folder structure:

```bash
MOT_root
├── train/
│   ├── MOT20-01/
│   │   ├── seqinfo.ini
│   │   ├── gt
│   │   │   ├── gt.txt
│   │   ├── img1
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   ├── MOT20-02/
│   │   ├── ...
├── test/
│   ├── ...
```

### KITTI Tracking

Folder structure:

```bash
KITTI_tracking_root
├── training/
│   ├── image_02/
│   │   ├── 0000
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── ...
│   │   ├── 0001
│   │   │   ├── ...
│   ├── label_02/
│   │   ├── 0000.txt
│   │   ├── 0001.txt
│   │   ├── ...
├── testing/
│   ├── ...
```

## Inference dataset

Typical usage

```python
from centernet_lightning.datasets import InferenceDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([A.Resize(width=512, height=512), A.Normalize(), ToTensorV2()])
dataset = InferenceDataset("img_folder", transforms=transforms)

# dataset[0] = {
#   "image_path": full path to the image
#   "image": image tensor in CHW format
#   "original_width": original image width, to convert bboxes if needed
#   "original_height": same as above
# }
```

Constructor

Argument | Description
---------|------------
`data_dir` | Path to data directory
`img_names` (default: `None`) | File names in `data_dir` to use. If `None`, all JPEG images in the folder will be used
`transforms` (default: `None`) | An `albumentations` transform 

Dataset item

Key | Description
----|------------
`image` | RGB image in `CHW` format
`image_path` | Full path to original image
`original_width` | Original image width
`original_height` | Original image height

Inference dataset does not need padding when batched. You can create a data loader directly from an inference dataset instance.

## Custom dataset for training

To write custom dataset, make sure you follow the format as described above.
