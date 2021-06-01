import os
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import COCODataset, collate_detections_with_padding, prepare_coco_detection
from model import ResNetBackbone, CenterNet

def get_train_augmentations(img_width=512, img_height=512):
    # from centernet paper
    # input is resized to 512x512
    # random flip, random scaling (0.6 - 1.3), color jittering (0.4)
    # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L222
    
    # use albumenations to take care of bbox transform
    # original centernet also uses PCA augmentation from cornernet, though it was not mentioned in their paper. The first PCA augmentation appeared in AlexNet https://dl.acm.org/doi/pdf/10.1145/3065386. Albumentations also implements this as FancyPCA
    # yolo bbox format is cxcywh
    train_augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(img_height, img_width),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=["labels"]))
    
    return train_augmentations

def train(config):
    # training hyperparameters
    num_epochs = config["TRAINER"]["EPOCHS"]
    batch_size = config["TRAINER"]["BATCH_SIZE"]
    lr = config["TRAINER"]["LEARNING_RATE"]

    # set up dataset
    train_data_dir = config["DATASET"]["TRAIN"]["DATA_DIR"]
    train_coco_version = config["DATASET"]["TRAIN"]["COCO_NAME"]
    val_data_dir = config["DATASET"]["VALIDATION"]["DATA_DIR"]
    val_coco_version = config["DATASET"]["VALIDATION"]["COCO_NAME"]

    prepare_coco_detection(train_data_dir, train_coco_version)
    prepare_coco_detection(val_data_dir, val_coco_version)

    train_augment = get_train_augmentations()
    train_dataset = COCODataset(train_data_dir, train_coco_version, transforms=train_augment)
    val_dataset = COCODataset(val_data_dir, val_coco_version)
    num_classes = train_dataset.num_classes

    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_detections_with_padding,
        batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_detections_with_padding,
        batch_size=batch_size, num_workers=4, pin_memory=True)

    # set up pytorch lightning model and trainer
    backbone_archi = config["MODEL"]["BACKBONE"]
    other_heads = config["MODEL"]["OTHER_HEADS"]

    backbone = ResNetBackbone(model=backbone_archi, pretrained=True)
    model = CenterNet(
        backbone=backbone, num_classes=num_classes, other_heads=other_heads,
        batch_size=batch_size, lr=lr)
    
    trainer = pl.Trainer(
        gpus=1, 
        # max_epochs=num_epochs,
        max_steps=500,              # train for 500 steps  
        limit_val_batches=20,       # only run validation on 20 batches
        val_check_interval=100,     # run validation every 100 steps
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    # load config from yaml file
    config_file = os.path.join("configs", "coco_resnet50.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train(config)
