import os
import yaml
from typing import Dict, Union
from enum import Enum
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from backbones import build_backbone
from losses import FocalLossWithLogits, render_target_heatmap
from metrics import class_tpfp_batch
from utils import convert_cxcywh_to_x1y1x2y2, draw_bboxes

_optimizer_mapper = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop
}

class Color(Enum):
    RED = (1., 0., 0.)
    BLUE = (0., 0., 1.)

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
        num_classes: int,
        backbone: Dict = None, 
        output_heads: Dict = None,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        **kwargs
        ):
        super(CenterNet, self).__init__()

        self.backbone = build_backbone(**backbone)        
        backbone_channels = self.backbone.out_channels
        
        self.num_classes = num_classes

        heatmap_bias = output_heads["heatmap_bias"]
        other_heads = output_heads["other_heads"]
        loss_weights = output_heads["loss_weights"]

        # for heatmap output, fill a pre-defined bias value
        # for other outputs, fill bias with 0 to match identity mapping (from centernet)
        self.output_heads = nn.ModuleDict()
        self.output_heads["heatmap"] = self._make_output_head(
            backbone_channels,
            num_classes,
            fill_bias=heatmap_bias
        )

        for h in other_heads:
            assert h in _supported_heads
            assert h in loss_weights
            self.output_heads[h] = self._make_output_head(
                backbone_channels,
                _output_head_channels[h],
                fill_bias=0
            )
        self.other_heads = other_heads
        self.loss_weights = loss_weights   
        
        # parameterized focal loss for heatmap
        self.focal_loss = FocalLossWithLogits(alpha=2., beta=4.)

        # get output stride from the backbone (how much input image is downsampled)
        self.output_stride = self.backbone.output_stride

        # for pytorch lightning tuner
        self.batch_size = batch_size
        self.learning_rate = learning_rate

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
        mask = targets["mask"].unsqueeze(-1)    # add column dimension to support broadcasting

        heatmap = output_maps["heatmap"]
        batch_size, _, out_h, out_w = heatmap.shape

        size_map = output_maps["size"].view(batch_size, 2, -1)      # flatten last xy dimensions
        offset_map = output_maps["offset"].view(batch_size, 2, -1)  # for torch.gather() later

        # initialize losses to 0
        losses = {}

        bboxes = bboxes.clone() / self.output_stride    # convert input coordinates to output coordinates
        centers_int = bboxes[...,:2].long()              # convert to integer to use as index

        # combine xy indices for torch.gather()
        # repeat indices using .expand() to gather on 2 channels
        xy_indices = centers_int[...,1] * out_w + centers_int[...,0]        # y * w + x
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size,2,-1))

        pred_sizes = torch.gather(size_map, dim=-1, index=xy_indices)       # N2D
        pred_offset = torch.gather(offset_map, dim=-1, index=xy_indices)    # N2D

        # need to swapaxes since pred_size is N2D but true_wh is ND2
        # use the mask to ignore none detections due to padding
        # NOTE: author noted that l1 loss is better than smooth l1 loss
        size_loss = F.l1_loss(pred_sizes.swapaxes(1,2), bboxes[...,2:], reduction="none")
        size_loss = torch.sum(size_loss * mask)
        losses["size"] = size_loss

        target_offset = bboxes[...,:2] - torch.floor(bboxes[...,:2])
        offset_loss = F.l1_loss(pred_offset.swapaxes(1,2), target_offset, reduction="none")
        offset_loss = torch.sum(offset_loss * mask)
        losses["offset"] = offset_loss

        losses["heatmap"] = torch.tensor(0., dtype=torch.float32, device=self.device)
        for b in range(batch_size):
            # render target heatmap and accumulate focal loss
            target_heatmap = render_target_heatmap(
                heatmap.shape[1:], centers_int[b], bboxes[b,:,:2], 
                labels[b], mask[b], device=self.device
            )
            losses["heatmap"] += self.focal_loss(heatmap[b], target_heatmap)

        # average over number of detections
        N = torch.sum(mask)
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
        heatmap = encoded_output["heatmap"]
        size_map = encoded_output["size"].view(batch_size, 2, -1)        # NCHW to NC(HW)
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
        topk_classes = topk_indices // (out_h*out_w)
        topk_xy_indices = topk_indices % (out_h*out_w)
        topk_y_indices = topk_xy_indices // out_w
        topk_x_indices = topk_xy_indices % out_w

        # extract bboxes at topk xy positions
        topk_x = topk_x_indices + torch.gather(offset_map[:,0], dim=-1, index=topk_xy_indices)
        topk_y = topk_y_indices + torch.gather(offset_map[:,1], dim=-1, index=topk_xy_indices)
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

        pred_detections = self.decode_detections(encoded_output, num_detections=50)
        class_tp, class_fp = self.evaluate_batch(pred_detections, batch)
        
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
        tp = np.zeros(self.num_classes, dtype=np.float32)
        fp = np.zeros(self.num_classes, dtype=np.float32)
        for x in outputs:
            tp += x["tp"]
            fp += x["fp"]
        
        precision = tp / (tp + fp + 1e-6)
        precision = np.average(precision)

        for i in range(precision.shape[0]):
            self.log(f"precision@50IoU/class_{i:02d}", precision[i])
        
        self.log("val/precision@50IoU_person", precision[0])
        self.log("val/precision@50IoU_car", precision[2])
        self.log("val/precision@50IoU", precision)

    @torch.no_grad()
    def evaluate_batch(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        # move to cpu and convert to numpy
        preds = {k: v.cpu().float().numpy() for k,v in preds.items()}
        targets = {k: v.cpu().float().numpy() for k,v in targets.items()}

        # convert cxcywh to x1y1x2y2
        convert_cxcywh_to_x1y1x2y2(preds["bboxes"])
        convert_cxcywh_to_x1y1x2y2(targets["bboxes"])

        class_tp, class_fp = class_tpfp_batch(preds, targets, self.num_classes, detection_threshold=0.1)
        return class_tp, class_fp

    @torch.no_grad()
    def draw_sample_images(
        self, imgs: torch.Tensor, preds: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor], N_samples: int=8
        ):
        N_samples = min(imgs.shape[0], N_samples)

        samples = imgs[:N_samples].cpu().float().numpy().transpose(0,2,3,1)    # convert NCHW to NHWC
        samples = np.ascontiguousarray(samples)     # C-contiguous, for opencv
        # samples = np.ascontiguousarray(samples[:,::2,::2,:])        # fast downsample via resampling  

        target_bboxes = targets["bboxes"].cpu().float().numpy()
        convert_cxcywh_to_x1y1x2y2(target_bboxes)
        target_labels = targets["labels"].cpu().numpy().astype(int)

        pred_bboxes = preds["bboxes"].cpu().float().numpy()
        convert_cxcywh_to_x1y1x2y2(pred_bboxes)
        pred_labels = preds["labels"].cpu().numpy().astype(int)
        pred_scores = preds["scores"].cpu().float().numpy()

        for i in range(N_samples):
            draw_bboxes(
                samples[i], pred_bboxes[i], pred_labels[i], pred_scores[i], 
                inplace=True, relative_scale=True, color=Color.RED)
    
            draw_bboxes(
                samples[i], target_bboxes[i], target_labels[i], 
                inplace=True, relative_scale=True, color=Color.BLUE)

        return samples

    def register_optimizer(self, optimizer_cfg: Dict):
        assert optimizer_cfg["name"] in _optimizer_mapper
        self.optimizer_cfg = optimizer_cfg

    # lightning method
    def configure_optimizers(self):
        if self.optimizer_cfg == None:
            warnings.warn("Optimizer config was not specified. Using adam optimizer with lr=1e-3")
            optimizer_params = dict(lr=1e-3)
            self.optimizer_cfg = dict(name="adam", params=optimizer_params)

        optimizer_algo = _optimizer_mapper[self.optimizer_cfg["name"]]
        optimizer = optimizer_algo(self.parameters(), **self.optimizer_cfg["params"])
        
        # lr scheduler
        return optimizer

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
