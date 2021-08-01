# Training CenterNet

It is recommended to train the model with the train script `train.py` to train with a config file.

```bash
python train.py --config "configs/coco_resnet34.yaml"
```

You can also import the `train()` function from the train script to train in your own script. You can either pass in path to your config file, or pass in a config dictionary directly.

```python
from train import train
from src.utils import load_config

# train with config file
train("config_file.yaml")

# train with config dictionary. you can modify dict values directly
config = load_config("config_file.yaml")
config["model"]["backbone"]["name"] = "resnet50"
config["trainer"]["max_epochs"] = 10
train(config)
```

The config file specifies everything required to train the model, including model construction, dataset, augmentations and training schedule.

## Custom model architecture

You can modify the backbone, neck, and output heads in their own section in the config file

## Custom dataset

Datasets in COCO and Pascal VOC formats are supported. See the Datasets section below to ensure your folder structure is correct. Change `data_dir` and `split` accordingly. For Pascal VOC, you also need to specify `name_to_label` to map class name to class label (number)

## Custom augmentations

Currently Albumentation is used to do augmentation. Any Albumentation transformations are supported. To specify a new augmentation, simply add to the list `transforms` under each dataset

## Custom trainer

This repo uses PyTorch Lightning, so we have all the PyTorch Lightning benefits. Specify any parameters you want to pass to the `trainer` in the config file to specify the training details. For a full list of option, refer to [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)

- Training epochs: Change `max_epochs`
- Multi-GPU training (not tested): Change `gpus`
- Mixed-precision training: Change `precision` to 16

## Custom optimizer and learning rate scheduler

Change `optimizer` and `lr_scheduler` under `model`. Only optimizers and schedulers from the official PyTorch is supported (in `torch.optim`). Not all schedulers will work, since they require extra information about training schedule. To use other optimizers, modify the `configure_optimizers()` method of the class `CenterNet`

## Manual training

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
