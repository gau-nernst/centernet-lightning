from enum import Enum
from typing import Dict
import numpy as np
import torch

import cv2
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

RED = (1., 0., 0.)
BLUE = (0., 0., 1.)

def convert_cxcywh_to_x1y1x2y2(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from cxcywh format to x1y2x2y2 format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2    # x1 = cx - w/2
    bboxes[...,1] -= bboxes[...,3] / 2    # y1 = cy - h/2
    bboxes[...,2] += bboxes[...,0]        # x2 = w + x1
    bboxes[...,3] += bboxes[...,1]        # y2 = h + y1

    return bboxes

def convert_cxcywh_to_xywh(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from cxcywh format to xywh format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()
    
    bboxes[...,0] -= bboxes[...,2] / 2      # x = cx - w/2
    bboxes[...,1] -= bboxes[...,3] / 2      # y = cy - h/2

    return bboxes

def convert_xywh_to_cxcywh(bboxes: np.ndarray, inplace=True):
    """Convert bboxes from xywh format to cxcywh format. Default is inplace
    """
    if not inplace:
        bboxes = bboxes.copy()

    bboxes[...,0] += bboxes[...,2] / 2      # cx = x + w/2
    bboxes[...,1] += bboxes[...,3] / 2      # cy = y + h/2

    return bboxes

def draw_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    inplace: bool = True,
    relative_scale: bool = False,
    color = (255,0,0)
    ):
    """Draw bounding boxes on an image using `cv2`
    
    Args:
        `img`: RGB image in HWC format. dtype is uint8 [0,255]
        `bboxes`: x1y1x2y2 format
    """
    if not inplace:
        img = img.copy()

    if relative_scale:
        bboxes = bboxes.copy()
        bboxes[:,[0,2]] *= img.shape[1]
        bboxes[:,[1,3]] *= img.shape[0]

    if type(scores) != np.ndarray:
        scores = np.ones_like(labels)

    for i in range(bboxes.shape[0]):
        pt1 = bboxes[i,:2].astype(int)
        pt2 = bboxes[i,2:].astype(int)
        text = f"{labels[i]} {scores[i]:.4f}"

        text_color = (0,0,0)
        cv2.rectangle(img, pt1, pt2, color, thickness=1)
        cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1)

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
    """input is 1-channel image with dimension NHW (no channel dimension)
    """
    cm = plt.get_cmap(cmap)
    output = cm(input)[...,:3]  # apply cmap and remove alpha channel

    if channel_first:
        output = output.transpose(0,3,1,2)  # NHWC to NCHW
    if return_tensor:
        output = torch.from_numpy(output)
    return output

class LogImageCallback(pl.Callback):
    def __init__(self, logger_type: bool = "tensorboard", num_samples: int = 8, num_bboxes: int = 10, cmap: str = "viridis"):
        assert logger_type in ("wandb", "tensorboard")
        
        self.logger_type = logger_type
        self.num_samples = num_samples      # number of images to draw
        self.num_bboxes  = num_bboxes        # number of bboxes to draw per image
        self.cmap        = cmap

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # only log sample images for the first validation batch
        if batch_idx == 0:
            # log images with predicted and target bboxes
            pred_detections = outputs["detections"]
            imgs = batch["image"]

            # only take the first num_bboxes of detections
            pred_detections   = {k: v[:,:self.num_bboxes].float().cpu().numpy() for k,v in pred_detections.items()}
            target_detections = {k: batch[k].float().cpu().numpy() for k in ["bboxes", "labels"]}
            convert_cxcywh_to_x1y1x2y2(pred_detections["bboxes"])
            convert_cxcywh_to_x1y1x2y2(target_detections["bboxes"])
            pred_detections["labels"]   = pred_detections["labels"].astype(int)
            target_detections["labels"] = target_detections["labels"].astype(int)

            # only take the first num_samples of images
            num_samples = min(imgs.shape[0], self.num_samples)
            imgs = imgs[:num_samples].float().cpu().numpy()
            imgs = imgs.transpose(0,2,3,1)      # NCHW to NHWC
            imgs = np.ascontiguousarray(imgs)   # for cv2
            
            for i in range(num_samples):
                draw_bboxes(
                    imgs[i],
                    pred_detections["bboxes"][i],
                    pred_detections["labels"][i],
                    pred_detections["scores"][i],
                    color=RED
                )
                draw_bboxes(
                    imgs[i],
                    target_detections["bboxes"][i],
                    target_detections["labels"][i],
                    color=BLUE
                )
            
            # log output heatmap and backbone output
            encoded_output = outputs["encoded_output"]
            pred_heatmap        = encoded_output["heatmap"][:self.num_samples].float().cpu()
            pred_heatmap        = torch.sigmoid(pred_heatmap)                   # convert to probability
            pred_heatmap, _     = torch.max(pred_heatmap, dim=1)                # max aggregate across classes/channels
            pred_heatmap_scaled = pred_heatmap / torch.max(pred_heatmap)        # scale to [0,1] for visualization
            pred_heatmap        = apply_mpl_cmap(pred_heatmap.numpy(), self.cmap)
            pred_heatmap_scaled = apply_mpl_cmap(pred_heatmap_scaled.numpy(), self.cmap)

            backbone_output = encoded_output["backbone_features"][:self.num_samples].float().cpu()
            backbone_output = torch.mean(backbone_output, dim=1)            # mean aggregate across channels
            backbone_output = apply_mpl_cmap(backbone_output.numpy(), self.cmap)

            log_images = {
                "output detections" : imgs,
                "heatmap"           : pred_heatmap,
                "heatmap (scaled)"  : pred_heatmap_scaled,
                "backbone output"   : backbone_output
            }
            
            for img_name, images in log_images.items():
                if self.logger_type == "wandb":
                    trainer.logger.experiment.log({
                        f"val/{img_name}": [wandb.Image(img) for img in images],
                        "global_step": trainer.global_step
                    })
                
                elif self.logger_type == "tensorboard":
                    trainer.logger.experiment.add_images(
                        f"val/{img_name}",
                        images, 
                        trainer.global_step,
                        dataformats="nhwc"
                    )
