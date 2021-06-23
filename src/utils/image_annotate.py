import warnings
from typing import Tuple, Iterable

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import Subset
import wandb

from ..datasets import IMAGENET_MEAN, IMAGENET_STD, build_dataset, render_target_heatmap_cornernet, render_target_heatmap_ttfnet
from .box import *

RED = (1., 0., 0.)
BLUE = (0., 0., 1.)

def draw_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    score_threshold: float = 0,
    inplace: bool = True,
    normalized_bbox: bool = False,
    color: Tuple[int] = (255,0,0),
    text_color: Tuple[int] = (0,0,0),
    font: int = cv2.FONT_HERSHEY_PLAIN
    ):
    """Draw bounding boxes on an image using `cv2`
    
    Args:
        `img`: an RGB image in HWC format, either in [0,255] or [0,1]
        `bboxes`: x1y1x2y2 format
        `labels`: class labels for each bbox
        `scores` (optional): confidence score to display with the label
        `score_threshold`: threshold to filter bboxes. Default is 0
        `inplace`: whether to draw bboxes directly on the original image or make a copy. Default is True
        `normalized_bbox`: whether the input bboxes are in normalized coordinates [0,1]. Default is False
        `color`: color used for bbox
        `text_color` and `font`: for text (label and score)
    """
    if not img.flags.c_contiguous:
        if inplace:
            warnings.warn("input image is not C-contiguous. this operation will not be inplace")
        img = np.ascontiguousarray(img)     # this will return a copy so inplace is ignored
    elif not inplace:
        img = img.copy()

    if normalized_bbox:
        bboxes = bboxes.copy()
        bboxes[:,[0,2]] *= img.shape[1]
        bboxes[:,[1,3]] *= img.shape[0]
    bboxes = bboxes.astype(int)

    for i in range(bboxes.shape[0]):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        pt1 = bboxes[i,:2]
        pt2 = bboxes[i,2:]
        text = f"{labels[i]}" if scores is None else f"{labels[i]} {scores[i]:.2f}"

        text_size = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_size)
        text_pt2 = (pt1[0] + text_width, pt1[1] - text_height)

        cv2.rectangle(img, pt1, pt2, color, thickness=1)                                # draw bbox
        cv2.rectangle(img, pt1, text_pt2, color, thickness=cv2.FILLED)                  # draw box for text label
        cv2.putText(img, text, pt1, font, text_size, text_color, thickness=text_size)   # draw text label

    return img

def draw_heatmap(img: np.ndarray, heatmap: np.ndarray, inplace: bool=True):
    """Draw heatmap on image. Both `img` and `heatmap` are in HWC format
    """
    if not inplace:
        img = img.copy()

    if heatmap.shape[-1] > 1:
        heatmap = np.max(heatmap, axis=-1)   # reduce to 1 channel

    # blend to first channel, using max
    img[:,:,0] = np.maximum(img[:,:,0], heatmap, out=img[:,:,0])
    return img

def apply_mpl_cmap(input: np.ndarray, cmap: str, return_tensor=False, channel_first=False):
    """input is a batch of 1-channel images. shape NHW (no channel dimension)
    """
    cm = plt.get_cmap(cmap)
    output = cm(input)[...,:3]  # apply cmap and remove alpha channel

    if channel_first:
        output = output.transpose(0,3,1,2)  # NHWC to NCHW
    if return_tensor:
        output = torch.from_numpy(output)
    return output

class LogImageCallback(pl.Callback):
    """Take a subset of `detection_file` based on the provided `indices`.
    """
    imagenet_mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    imagenet_std  = np.array(IMAGENET_STD, dtype=np.float32)
    cmap = "viridis"

    def __init__(self, dataset_cfg, indices = None):
        if indices is None:
            indices = range(16)
        elif isinstance(indices, int):
            indices = range(indices)
        
        dataset = build_dataset(**dataset_cfg)
        dataset = Subset(dataset, indices)
        self.dataset = dataset

    # log target heatmap on fit start
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get heatmap shape and allocate memory
        img_shape = self.dataset[0]["image"].shape
        heatmap_shape = (pl_module.num_classes, img_shape[1]//pl_module.output_stride, img_shape[2]//pl_module.output_stride)
        heatmaps = [np.zeros(heatmap_shape) for _ in range(len(self.dataset))]

        # render target heatmap
        for i, item in enumerate(self.dataset):
            if pl_module.hparams["heatmap_method"] == "cornernet":
                render_target_heatmap_cornernet(heatmaps[i], item["bboxes"], item["labels"])
            elif pl_module.hparams["heatmap_method"] == "ttfnet":
                render_target_heatmap_ttfnet(heatmaps[i], item["bboxes"], item["labels"])

            heatmaps[i] = np.max(heatmaps[i], axis=0)
            heatmaps[i] = apply_mpl_cmap(heatmaps[i], self.cmap)

        # make into a grid and log the image
        heatmap_grid = make_image_grid(heatmaps)
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({"target heatmap": wandb.Image(heatmap_grid)})
        elif isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image("target heatmap", heatmap_grid, dataformats="hwc")

    # run inference and log predicted detections
    @torch.no_grad()
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        log_images = {
            "heatmap output": [],
            "heatmap (scaled)": [],
            "backbone output": []
        }
        images = []
        detections_target = {
            "bboxes": [],
            "labels": []
        }
        detections_pred = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }
        _, img_height, img_width = self.dataset[0]["image"].shape

        pl_module.eval()
        for item in self.dataset:
            # save image to make a grid later
            img = item["image"]
            img_np = img.clone().numpy().transpose(1,2,0) * self.imagenet_std + self.imagenet_mean
            images.append(img_np)
            
            # save ground truth bboxes and labels
            for k in detections_target.keys():
                detections_target[k].append(np.array(item[k]))
            
            detections_target["bboxes"][-1][...,[0,2]] *= img_width
            detections_target["bboxes"][-1][...,[1,3]] *= img_height
            convert_cxcywh_to_x1y1x2y2(detections_target["bboxes"][-1])

            # only key "image" is need to run inference
            img = {"image": img.unsqueeze(0).to(pl_module.device)}
            encoded_outputs = pl_module(img)
            pred_detections = pl_module.decode_detections(encoded_outputs)

            for k in detections_pred.keys():
                detections_pred[k].append(pred_detections[k][0].cpu().numpy())
            convert_cxcywh_to_x1y1x2y2(detections_pred["bboxes"][-1])

            # log heatmap output
            heatmap_output = encoded_outputs["heatmap"][0].cpu().float()    # 80 x 128 x 128
            heatmap_output, _ = torch.max(heatmap_output, dim=0)            # 128 x 128
            heatmap_output = torch.sigmoid(heatmap_output)
            heatmap_scaled = heatmap_output / torch.max(heatmap_output)

            heatmap_output = apply_mpl_cmap(heatmap_output.numpy(), self.cmap)  # 128 x 128 x 3
            log_images["heatmap output"].append(heatmap_output)

            heatmap_scaled = apply_mpl_cmap(heatmap_scaled.numpy(), self.cmap)
            log_images["heatmap (scaled)"].append(heatmap_scaled)

            # log backbone output
            backbone_output = encoded_outputs["backbone_features"][0].cpu().float()
            backbone_output = torch.mean(backbone_output, dim=0)
            backbone_output = apply_mpl_cmap(backbone_output.numpy(), self.cmap)
            log_images["backbone output"].append(backbone_output)

        img_grid, target_bboxes, pred_bboxes = make_image_grid(images, detections_target["bboxes"], detections_pred["bboxes"])
        pred_labels = np.concatenate(detections_pred["labels"], axis=0)
        pred_scores = np.concatenate(detections_pred["scores"], axis=0)
        target_labels = np.concatenate(detections_target["labels"], axis=0)

        log_images = {k: make_image_grid(v) for k,v in log_images.items()}

        if isinstance(trainer.logger, WandbLogger):
            wandb_log = {
                "detections": wandb.Image(img_grid, boxes={
                    "predictions": {"box_data": convert_bboxes_to_wandb(pred_bboxes, pred_labels, pred_scores)},
                    "ground_truth": {"box_data": convert_bboxes_to_wandb(target_bboxes, target_labels, np.ones(len(target_labels)))}
                }),
                "global_step": trainer.global_step
            }

            for name, img in log_images.items():
                wandb_log[name] = wandb.Image(img)
            
            trainer.logger.experiment.log(wandb_log)

        elif isinstance(trainer.logger, TensorBoardLogger):
            draw_bboxes(img_grid, pred_bboxes, pred_labels, pred_scores, color=RED)
            draw_bboxes(img_grid, target_bboxes, target_labels, color=BLUE)
            trainer.logger.experiment.add_image("detections", img_grid, trainer.global_step, dataformats="hwc")

            for name, img in log_images.items():
                trainer.logger.experiment.add_image(name, img, trainer.global_step, dataformats="hwc")

def make_image_grid(imgs: Iterable[np.ndarray], bboxes1: Iterable[np.ndarray] = None, bboxes2: Iterable[np.ndarray] = None, imgs_per_row: int = 8):
    """
    Args
        imgs: a list of images in HWC format
        bboxes1 and bboxes2 (optional): a list of bboxes. intended for ground truth boxes and predicted boxes
    """
    num_imgs = len(imgs)
    img_height, img_width, channels = imgs[0].shape
    if bboxes1 is not None:
        bboxes1 = [bboxes.copy() for bboxes in bboxes1]
    if bboxes2 is not None:
        bboxes2 = [bboxes.copy() for bboxes in bboxes2]

    num_rows = np.ceil(num_imgs / imgs_per_row).astype(int)
    grid = np.zeros((num_rows*img_height, imgs_per_row*img_width, channels), dtype=imgs[0].dtype)

    for i, img in enumerate(imgs):
        y = i // imgs_per_row
        x = i % imgs_per_row
        grid[y*img_height:(y+1)*img_height, x*img_width:(x+1)*img_width] = img
        
        # shift the bboxes
        if bboxes1 is not None:
            bboxes1[i][:,[0,2]] += x * img_width
            bboxes1[i][:,[1,3]] += y * img_height
        if bboxes2 is not None:
            bboxes2[i][:,[0,2]] += x * img_width
            bboxes2[i][:,[1,3]] += y * img_height

    # combine bboxes of several images into a single list of bboxes
    if bboxes1 is not None:
        bboxes1 = np.concatenate(bboxes1, axis=0)
    if bboxes2 is not None:
        bboxes2 = np.concatenate(bboxes2, axis=0)

    if bboxes1 is not None:
        if bboxes2 is not None:
            return grid, bboxes1, bboxes2
        return grid, bboxes1
    return grid

def convert_bboxes_to_wandb(bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray = None):
    """
    """
    wandb_boxes = []
    bboxes = bboxes.astype(int)
    labels = labels.astype(int)

    for i in range(len(labels)):
        item = {
            "position": {
                "minX": bboxes[i][0].item(),
                "minY": bboxes[i][1].item(),
                "maxX": bboxes[i][2].item(),
                "maxY": bboxes[i][3].item()
            },
            "domain": "pixel",
            "class_id": labels[i].item()
        }
        if scores is not None:
            item["scores"] = {"confidence": scores[i].item()}
        
        wandb_boxes.append(item)
    
    return wandb_boxes
