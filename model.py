import os
import yaml
from typing import Dict, Union
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from backbones import build_backbone
from losses import FocalLossWithLogits, render_target_heatmap_ttfnet, render_target_heatmap_cornernet
from metrics import class_tpfp_batch
from utils import convert_cxcywh_to_x1y1x2y2

_supported_heads = ["size", "offset"]
_output_head_channels = {
    "size": 2,
    "offset": 2
}

class CenterNet(pl.LightningModule):
    """General CenterNet model. Build CenterNet from a given backbone and output
    """
    def __init__(
        self,
        backbone: Dict, 
        num_classes: int,
        output_heads: Dict = None,
        optimizer: Dict = None,
        lr_scheduler: Dict = None,
        **kwargs
        ):
        super(CenterNet, self).__init__()

        self.backbone = build_backbone(**backbone)        
        backbone_channels = self.backbone.out_channels
        self.output_stride = self.backbone.output_stride    # how much input image is downsampled
        
        self.num_classes = num_classes

        heatmap_bias = output_heads["heatmap_bias"]
        other_heads  = output_heads["other_heads"]
        loss_weights = output_heads["loss_weights"]
        loss_functions = output_heads["loss_functions"]

        # create output heads and set their losses
        self.output_heads = nn.ModuleDict()
        self.head_loss_fn = nn.ModuleDict()
        self.output_heads["heatmap"] = self._make_output_head(
            backbone_channels,
            num_classes,
            fill_bias=heatmap_bias          # use heatmap_bias for heatmap output
        )
        self.head_loss_fn["heatmap"] = FocalLossWithLogits(alpha=2., beta=4.)       # focal loss for heatmap

        for h in other_heads:
            assert h in _supported_heads
            assert h in loss_weights
            assert h in loss_functions
            head = self._make_output_head(
                backbone_channels,
                _output_head_channels[h],
                fill_bias=0                 # set bias to 0
            )
            # loss for size and offset head should be either L1Loss or SmoothL1Loss
            # cornernet uses smooth l1 loss, centernet uses l1 loss
            # NOTE: centernet author noted that l1 loss is better than smooth l1 loss
            loss_fn = nn.__dict__[loss_functions[h]](reduction=None)    # don't use reduction to apply mask later
            self.output_heads[h] = head
            self.head_loss_fn[h] = loss_fn
        
        self.other_heads  = other_heads
        self.loss_weights = loss_weights   
        
        self.optimizer_cfg    = optimizer
        self.lr_scheduler_cfg = lr_scheduler

        self.save_hyperparameters()
        self._steps_per_epoch = None        # for steps_per_epoch property

    def _make_output_head(self, in_channels: int, out_channels: int, fill_bias: float = None):
        # Reference implementations
        # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L125    use num_filters = 256
        # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py#L5      use num_filters = in_channels
        conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels, out_channels, 1)

        if fill_bias != None:
            conv2.bias.data.fill_(fill_bias)

        output_head = nn.Sequential(conv1, relu, conv2)
        return output_head

    def forward(self, batch):
        """Return a dictionary of feature maps for each output head. Use this output to either decode to predictions or compute loss.
        """
        img = batch["image"]

        features = self.backbone(img)
        output = {}
        output["backbone_features"] = features      # for logging purpose
        
        for k, v in self.output_heads.items():
            output[k] = v(features)
        
        return output

    def compute_loss(self, output_maps: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Return a dictionary of losses for each output head. This method is called during the training step
        """
        bboxes = targets["bboxes"]              # ND4
        labels = targets["labels"]              # ND
        mask   = targets["mask"].unsqueeze(-1)  # add column dimension to support broadcasting

        heatmap = output_maps["heatmap"]
        batch_size, _, out_h, out_w = heatmap.shape

        size_map = output_maps["size"].view(batch_size, 2, -1)      # flatten last xy dimensions
        offset_map = output_maps["offset"].view(batch_size, 2, -1)  # for torch.gather() later

        # initialize losses to 0
        losses = {}

        out_bboxes = bboxes / self.output_stride    # convert input image coordinates to output image coordinates
        centers_int = out_bboxes[...,:2].long()     # convert to integer to use as index

        # combine xy indices for torch.gather()
        # repeat indices using .expand() to gather on 2 channels
        xy_indices = centers_int[...,1] * out_w + centers_int[...,0]        # y * w + x
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size,2,-1))

        pred_sizes  = torch.gather(size_map, dim=-1, index=xy_indices)      # N2D
        pred_offset = torch.gather(offset_map, dim=-1, index=xy_indices)    # N2D

        # need to swapaxes since pred_size is N2D but true_wh is ND2
        size_loss = self.head_loss_fn["size"](pred_sizes.swapaxes(1,2), out_bboxes[...,2:])
        size_loss = torch.sum(size_loss * mask)
        losses["size"] = size_loss

        target_offset = out_bboxes[...,:2] - torch.floor(out_bboxes[...,:2])
        offset_loss = self.head_loss_fn["offset"](pred_offset.swapaxes(1,2), target_offset)
        offset_loss = torch.sum(offset_loss * mask)
        losses["offset"] = offset_loss

        losses["heatmap"] = torch.tensor(0., dtype=torch.float32, device=self.device)
        for b in range(batch_size):
            # render target heatmap and accumulate focal loss
            # 2 versions: cornernet (original) and ttfnet
            target_heatmap = render_target_heatmap_cornernet(
                heatmap.shape[1:],
                out_bboxes[b],
                labels[b],
                mask[b],
                device=self.device
            )
            losses["heatmap"] += self.head_loss_fn["heatmap"](heatmap[b], target_heatmap)

        # average over number of detections
        N = torch.sum(mask) + 1e-8
        losses["heatmap"] /= N
        losses["size"] /= N
        losses["offset"] /= N

        return losses

    def decode_detections(self, encoded_output: Dict[str, torch.Tensor], num_detections: int = 100, nms_kernel: int = 3):
        """Decode model output to detections
        """
        # reference implementations
        # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L234
        # https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L118
        # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_decode.py#L28
        batch_size, _, out_h, out_w = encoded_output["heatmap"].shape
        heatmap    = encoded_output["heatmap"]
        size_map   = encoded_output["size"].view(batch_size, 2, -1)        # NCHW to NC(HW)
        offset_map = encoded_output["offset"].view(batch_size, 2, -1)

        # obtain topk from heatmap
        # heatmap = torch.sigmoid(encoded_output["heatmap"])  # convert to probability NOTE: is this necessary? sigmoid is a monotonic increasing function. max order will be preserved
        
        local_peaks = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=(nms_kernel-1)//2)
        nms_mask = (heatmap == local_peaks)  # pseudo-nms, only consider local peaks
        heatmap = nms_mask.float() * heatmap

        # flatten to N(CHW) to apply topk
        heatmap = heatmap.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(heatmap, num_detections)
        topk_scores = torch.sigmoid(topk_scores)

        # restore flattened indices to class, xy indices
        topk_classes    = topk_indices // (out_h*out_w)
        topk_xy_indices = topk_indices % (out_h*out_w)
        topk_y_indices  = topk_xy_indices // out_w
        topk_x_indices  = topk_xy_indices % out_w

        # extract bboxes at topk xy positions
        # they are in input image coordinates (512x512)
        topk_x = (topk_x_indices + torch.gather(offset_map[:,0], dim=-1, index=topk_xy_indices)) * self.output_stride
        topk_y = (topk_y_indices + torch.gather(offset_map[:,1], dim=-1, index=topk_xy_indices)) * self.output_stride
        topk_w = torch.gather(size_map[:,0], dim=-1, index=topk_xy_indices)
        topk_h = torch.gather(size_map[:,1], dim=-1, index=topk_xy_indices)

        topk_bboxes = torch.stack([topk_x, topk_y, topk_w, topk_h], dim=-1)  # NK4
        out = {
            "labels": topk_classes,
            "bboxes": topk_bboxes,
            "scores": topk_scores
        }
        return out

    # lightning method, return total loss here
    def training_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)
        
        total_loss = losses["heatmap"]
        for h in self.other_heads:
            total_loss += losses[h] * self.loss_weights[h]

        for k,v in losses.items():
            self.log(f"train/{k}_loss", v)
        self.log("train/total_loss", total_loss)
        self.log("epoch_frac", self.global_step / self.steps_per_epoch)     # log this to view graph with epoch as x-axis

        return total_loss

    def validation_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)

        total_loss = losses["heatmap"]
        for h in self.other_heads:
            total_loss += losses[h] * self.loss_weights[h]

        for k,v in losses.items():
            self.log(f"val/{k}_loss", v)
        self.log("val/total_loss", total_loss)

        pred_detections = self.decode_detections(encoded_output, num_detections=100)

        class_tp = np.zeros((11, self.num_classes))
        class_fp = np.zeros((11, self.num_classes))

        for i in range(11):
            # detection thresholds 0.0, 0.1, ..., 1.0
            tp, fp = self.evaluate_batch(pred_detections, batch, detection_threshold=i/10)
            class_tp[i] = tp
            class_fp[i] = fp
        
        result = {
            "tp": class_tp,
            "fp": class_fp,
        }

        # return these for logging image callback
        if batch_idx == 0:
            result["detections"] = pred_detections
            result["encoded_output"] = encoded_output
    
        return result

    def validation_epoch_end(self, outputs):
        class_tp = np.zeros((11, self.num_classes), dtype=np.float32)
        class_fp = np.zeros((11, self.num_classes), dtype=np.float32)
        for batch in outputs:
            class_tp += batch["tp"]
            class_fp += batch["fp"]
        
        precision = class_tp / (class_tp + class_fp + 1e-8)
        ap = np.average(precision, axis=0)      # average over detection thresholds to get AP
        mAP = np.average(ap)                    # average over classes to get mAP

        for i, class_ap in enumerate(ap):
            self.log(f"AP50/class_{i:02d}", class_ap)
        
        self.log("val/mAP50", mAP)
        self.log("val/AP50_person", ap[1])
        self.log("val/AP50_car", ap[3])

    @torch.no_grad()
    def evaluate_batch(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], detection_threshold: float = 0.5):
        # move to cpu and convert to numpy
        preds = {k: v.cpu().float().numpy() for k,v in preds.items()}
        targets = {k: v.cpu().float().numpy() for k,v in targets.items()}

        # convert cxcywh to x1y1x2y2
        convert_cxcywh_to_x1y1x2y2(preds["bboxes"])
        convert_cxcywh_to_x1y1x2y2(targets["bboxes"])

        class_tp, class_fp = class_tpfp_batch(preds, targets, self.num_classes, detection_threshold=detection_threshold)
        return class_tp, class_fp

    @property
    def steps_per_epoch(self):
        # does not consider multi-gpu training
        if self.trainer.max_steps:
            return self.trainer.max_steps
        
        if self._steps_per_epoch is None:
            self._steps_per_epoch = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        
        return self._steps_per_epoch

    def register_optimizer(self, optimizer_cfg: Dict):
        self.optimizer_cfg = optimizer_cfg

    def register_lr_scheduler(self, lr_scheduler_cfg: Dict):
        self.lr_scheduler_cfg = lr_scheduler_cfg

    # lightning method
    def configure_optimizers(self):
        if self.optimizer_cfg == None:
            warnings.warn("Optimizer config was not specified. Using adam optimizer with lr=1e-3")
            optimizer_params = dict(lr=1e-3)
            self.optimizer_cfg = dict(name="Adam", params=optimizer_params)

        optimizer_algo = torch.optim.__dict__[self.optimizer_cfg["name"]]
        optimizer = optimizer_algo(self.parameters(), **self.optimizer_cfg["params"])

        if self.lr_scheduler_cfg is None:
            return optimizer
        
        # NOTE: need a better way to manage lr_schedulers
        # - some lr schedulers need info about training -> cannot provide from config file e.g. OneCycleLR
        # - support for multiple lr schedulers
        lr_scheduler_algo = torch.optim.lr_scheduler.__dict__[self.lr_scheduler_cfg["name"]]
        if self.lr_scheduler_cfg["name"] in ("OneCycleLR"):
            lr_scheduler = lr_scheduler_algo(optimizer, epochs=self.trainer.max_epochs, steps_per_epoch=self.steps_per_epoch, **self.lr_scheduler_cfg["params"])
        else:
            lr_scheduler = lr_scheduler_algo(optimizer, **self.lr_scheduler_cfg["params"])

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

def build_centernet_from_cfg(cfg_file: Union[str, Dict]):
    """Build CenterNet from a confile file.

    Args:
        cfg_file (str or dict): Config file to build CenterNet, either path to the config file, or a config file dictionary
    """
    if type(cfg_file) == str:
        assert os.path.exists(cfg_file), "Config file does not exist"
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            params = config["model"]

    else:
        params = cfg_file

    model = CenterNet(**params)
    return model
