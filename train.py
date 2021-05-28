import os
import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl

from datasets import COCODataset, collate_bbox_labels
from model import ResNetBackbone, CenterNet

def get_train_augmentations(img_width=512, img_height=512):
    # from centernet paper
    # input is resized to 512x512
    # random flip, random scaling (0.6 - 1.3), color jittering (0.4)
    # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L222
    
    # use albumenations to take care of bbox transform
    # original centernet also uses PCA augmentation from cornernet, though it was not mentioned in their paper. The first PCA augmentation appeared in AlexNet https://dl.acm.org/doi/pdf/10.1145/3065386. Albumentations also implements this as FancyPCA
    # yolo bbox format is center xy wh
    train_augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(img_height, img_width),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=["labels"]))
    
    return train_augmentations

def train():
    # training hyperparameters
    num_epochs = 1
    batch_size = 16
    lr = 1e-4

    # set up dataset
    # data_dir = os.path.join(os.environ["HOME"], "thien", "datasets", "COCO")
    data_dir = "D://datasets/COCO"
    coco_version = "val2017"
    
    train_augment = get_train_augmentations()
    coco_dataset = COCODataset(data_dir, coco_version, transforms=train_augment)
    num_classes = coco_dataset.num_classes

    train_dataloader = DataLoader(coco_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(coco_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # set up pytorch lightning model and trainer
    backbone = ResNetBackbone(model="resnet50", pretrained=True)
    model = CenterNet(backbone=backbone, num_classes=num_classes, batch_size=batch_size, lr=lr)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs)

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    train()