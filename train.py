import os
import yaml

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import COCODataset, collate_detections_with_padding, prepare_coco_detection
from model import simple_mobilenet_backbone, simple_resnet_backbone, fpn_resnet_backbone, fpn_mobilenet_backbone, CenterNet

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

def apply_mpl_cmap(input: torch.Tensor, cmap: str, return_tensor=False, channel_first=False):
    """input is 1-channel image with dimension NHW (no channel dimension)
    """
    cm = plt.get_cmap(cmap)
    output = cm(input.numpy())[...,:3]  # apply cmap and remove alpha channel

    if channel_first:
        output = output.transpose(0,3,1,2)  # NHWC to NCHW
    if return_tensor:
        output = torch.from_numpy(output)
    return output

class LogImageCallback(pl.Callback):
    
    def __init__(self, use_wandb=False, num_samples=8, num_bboxes=10):
        self.use_wandb = use_wandb
        self.num_samples = num_samples
        self.num_bboxes = num_bboxes

    def on_validation_batch_end(self, trainer, pl_module: CenterNet, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            pred_detections = outputs["detections"]
            encoded_output = outputs["encoded_output"]

            pred_detections = {k: v[:,:self.num_bboxes] for k,v in pred_detections.items()}

            # only log sample images for the first validation batch
            imgs = batch["image"]
            cmap = "viridis"

            # draw bounding boxes on val images
            sample_imgs = pl_module.draw_sample_images(imgs, pred_detections, batch, N_samples=self.num_samples)
            
            # log output heatmap
            pred_heatmap = encoded_output["heatmap"][:self.num_samples].cpu()
            pred_heatmap = torch.sigmoid(pred_heatmap)                      # convert to probability
            pred_heatmap, _ = torch.max(pred_heatmap, dim=1)                # aggregate heatmaps across classes/channels
            pred_heatmap_scaled = pred_heatmap / torch.max(pred_heatmap)    # scale to [0,1]

            pred_heatmap = apply_mpl_cmap(pred_heatmap, cmap)
            pred_heatmap_scaled = apply_mpl_cmap(pred_heatmap_scaled, cmap)

            # log backbone feature map
            backbone_feature_map = encoded_output["backbone_features"][:self.num_samples].cpu()
            backbone_feature_map = torch.mean(backbone_feature_map, dim=1)      # mean aggregate
            backbone_feature_map = apply_mpl_cmap(backbone_feature_map, cmap)
            
            log_images = {
                "output detections": sample_imgs,
                "predicted heatmap": pred_heatmap,
                "predicted heatmap scaled": pred_heatmap_scaled,
                "backbone feature map": backbone_feature_map
            }
            
            for img_name, imgs in log_images.items():
                if self.use_wandb:  # log using wandb
                    trainer.logger.experiment.log({
                        f"val/{img_name}": [wandb.Image(img) for img in imgs],
                        "global_step": trainer.global_step
                    })
                
                else:               # log using tensorboard
                    trainer.logger.experiment.add_images(
                        f"val/{img_name}", imgs, 
                        trainer.global_step, dataformats="nhwc")


def train(config, run_name=None, use_wandb=False):
    # training hyperparameters
    num_epochs = config["TRAINER"]["EPOCHS"]
    batch_size = config["TRAINER"]["BATCH_SIZE"]
    optimizer = config["TRAINER"]["OPTIMIZER"]
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
        batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_detections_with_padding,
        batch_size=batch_size, num_workers=4, pin_memory=True)

    # set up pytorch lightning model and trainer
    backbone_family = config["MODEL"]["BACKBONE"]["FAMILY"]
    backbone_fpn = config["MODEL"]["BACKBONE"]["FPN"]
    backbone_archi = config["MODEL"]["BACKBONE"]["ARCHITECTURE"]
    upsample_init = config["MODEL"]["BACKBONE"]["UPSAMPLE_INIT_BILINEAR"]
    other_heads = config["MODEL"]["OUTPUT_HEADS"]["OTHER_HEADS"]
    heatmap_bias = config["MODEL"]["OUTPUT_HEADS"]["HEATMAP_BIAS"]
    loss_weights = config["MODEL"]["OUTPUT_HEADS"]["LOSS_WEIGHTS"]
    loss_weights = {k.lower(): v for k,v in loss_weights.items()}

    # build model
    if backbone_family == "resnet":
        if backbone_fpn:
            backbone = fpn_resnet_backbone(backbone_archi, pretrained=True)
        else:
            backbone = simple_resnet_backbone(
                backbone_archi, pretrained=True, upsample_init_bilinear=upsample_init)
    
    elif backbone_family == "mobilenet":
        if backbone_fpn:
            backbone = fpn_mobilenet_backbone(backbone_archi, pretrained=True)
        else:
            backbone = simple_mobilenet_backbone(
                backbone_archi, pretrained=True, upsample_init_bilinear=upsample_init)
    else:
        raise ValueError(f"{backbone_family} not supported")
    
    model = CenterNet(
        backbone=backbone, num_classes=num_classes, other_heads=other_heads,
        heatmap_bias=heatmap_bias, loss_weights=loss_weights,
        batch_size=batch_size, optimizer=optimizer, lr=lr)
    
    if use_wandb:
        logger = WandbLogger(project="CenterNet", name=run_name, log_model=True)
        logger.watch(model)
    else:
        logger = TensorBoardLogger("tb_logs", name=run_name)
    
    logger.log_hyperparams({
        "backbone_architecture": backbone_archi,
        "upsample_init_bilinear": upsample_init,
        "heatmap_bias": heatmap_bias,
        "size_loss_weight": loss_weights["size"],
        "offset_loss_weight": loss_weights["offset"],
        "batch_size": batch_size,
        "optimizer": optimizer,
        "learning_rate": lr
    })

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=num_epochs,
        # max_steps=500,              # train for 500 steps  
        limit_val_batches=50,       # only run validation on 20 batches
        val_check_interval=1000,     # run validation every 100 steps
        logger=logger,
        callbacks=[LogImageCallback(use_wandb, 16)]
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    run_name = "coco_resnet34_1epoch_test"
    use_wandb = True

    if use_wandb:
        # read wandb API key from file
        with open(".wandb_key", "r", encoding="utf-8") as f:
            os.environ["WANDB_API_KEY"] = f.readline().rstrip()
        
    # load config from yaml file
    config_file = os.path.join("configs", "coco_resnet34.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train(config, run_name=run_name, use_wandb=use_wandb)
