# Installation

Main dependencies

- `pytorch`
- `torchvision`
- `numpy`
- `filterpy` (for Kalman filter in tracking)
- `opencv-python`
- `pytorch-lightning`
- `albumentations>=1.0.0` (for augmentations and image transformations)
- `cython` (for `pycocotools`)
- `pycocotools` (to read COCO dataset and evaluate detection. Use [gautamchitnis](https://github.com/gautamchitnis/cocoapi) fork to support Windows)
- `trackeval` (to evaluate tracking. Until the author adds an installation script, use [my fork](https://github.com/gau-nernst/TrackEval) to install it as package)

Other dependencies

- `ipykernel` (to use with Jupyter)
- `pytest` (for unit testing, not required to run)
- `wandb` (for Weights and Biases logging, not required to run)
- `ffmpeg` (for converting tracking frames to a video)

Environment tested: Windows 10 and Linux (Ubuntu), `python=3.8`, `pytorch=1.8.1`, `torchvision=0.9.1`, `cudatoolkit=11.1`. Any recent versions of PyTorch should work.

Note on ffmpeg: When installing `torchvision` via `conda`, `ffmpeg` is also installed depending on `torchvision` version. However, this `ffmpeg` is not shipped with x264 encoder. You can either run `conda install ffmpeg` to install the missing encoders, or use a separate ffmpeg binary.

## Preparation

Clone this repo and navigate to the repo directory

```bash
git clone <THIS_REPO_GIT_URL>
cd CenterNet
```

## Install from `environment.yml`

```bash
conda env create -f environment.yml
conda activate centernet
```

## Manual install

Create a new virtual environment

```bash
conda env create -n centernet python=3.8
conda activate centernet
```

Install PyTorch. Follow the official installation instruction [here](https://pytorch.org/)

```bash
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
```

If you don't have NVIDIA GPU or don't need GPU support, remove `cudatoolkit=11.1` and `-c conda-forge`.

Install `pip` dependencies

```bash
pip install filterpy cython pytorch-lightning opencv-python albumentations
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
pip install git+https://github.com/gau-nernst/TrackEval.git

# optional packages
pip install ipykernel pytest wandb
```
