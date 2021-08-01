# Datasets

The following dataset formats are supported:

Detection:

- [x] (COCO)[https://cocodataset.org/]
- [x] (Pascal VOC)[http://host.robots.ox.ac.uk/pascal/VOC/]
- [x] (CrowdHuman)[https://www.crowdhuman.org/]

Tracking:

- [x] (MOT)[https://motchallenge.net/]
- [x] (KITTI Tracking)[http://www.cvlibs.net/datasets/kitti/eval_tracking.php]

## Usage

WIP

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
from src.datasets import COCODataset

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

Make sure your file extension is correct (`.jpg` for images, `.xml` for annotations, and `.txt` for image sets/splits).

The dataset split (e.g. `train.txt`) contains a list of image names without file extension. Each name is on a separate line.

To create a Pascal VOC dataset

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

## Custom dataset for training

You can write your own custom dataset, as long as it conforms to the format that the model expects. A batch of input data should be a dictionary with the following key-value pairs:

- `image`: images in `CHW` format. Shape `NCHW`.
- `bboxes`: bounding boxes in `(cx,cy,w,h)` format (unit: pixel). Shape `ND4`, where `D` is the number of detections in one image.
- `labels`: labels `[0,num_classes-1]`. Shape `ND`.
- `mask`: binary mask of `0` or `1`. Since each image has different number of detections, `bboxes` and `labels` are padded so that they have the same lengths within one batch. This is used in calculating loss. Shape `ND`.

If you only need inference, only key `image` is needed.
