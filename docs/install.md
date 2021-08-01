# Installation

Main dependencies

- pytorch, torchvision
- numpy
- opencv-python
- pytorch-lightning
- pycocotools (to read COCO dataset and evaluate detection. Cython is required. Use [gautamchitnis](https://github.com/gautamchitnis/cocoapi) fork to support Windows)
- albumentations (for augmentations during training)
- trackeval (to evaluate tracking. Until the author adds an installation script, use [my fork](https://github.com/gau-nernst/TrackEval) to install it as package)

Other dependencies

- ipykernel (to use with Jupyter)
- pytest (for unit testing, not required to run)
- wandb (for Weights and Biases logging, not required to run)
- ffmpeg (for converting tracking frames to a video)

Environment tested: Windows 10 and Linux (Ubuntu), python=3.8, pytorch=1.8.1, torchvision=0.9.1, cudatoolkit=11.1

Note on albumentations: please use `albumentations >= 1.0.0`. There are some problems with YOLO box format in previous versions.

Note on ffmpeg: When installing torchvision via conda, ffmpeg is also installed depending on torchvision version. However, this ffmpeg is not shipped with x264 encoder. You can either run `conda install ffmpeg` to install the missing encoders, or use a separate ffmpeg binary.

## Install with conda

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
pip install cython pytorch-lightning opencv-python
pip install -U albumentations --no-binary imgaug,albumentations
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
pip install git+https://github.com/gau-nernst/TrackEval.git

# optional packages
pip install ipykernel pytest wandb
```
