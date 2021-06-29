from typing import Dict
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb

from ..backbones import build_backbone
from ..losses import focal_loss, iou_loss
from ..datasets import InferenceDataset
from ..utils import convert_cxcywh_to_xywh
from ..eval import detections_to_coco_results

__all__ = ["CenterNet"]

_supported_heads = ["size", "offset"]
_output_head_channels = {
    "size": 2,
    "offset": 2
}
_supported_losses = {
    "modified_focal": focal_loss.ModifiedFocalLossWithLogits,
    "quality_focal": focal_loss.QualityFocalLossWithLogits,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "iou": iou_loss.CenterNetIoULoss,
    "giou": iou_loss.CenterNetGIoULoss
}

def _make_output_head(in_channels: int, hidden_channels: int, out_channels: int, fill_bias: float = None):
    # Reference implementations
    # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L125    use num_filters = 256
    # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py#L5      use num_filters = in_channels
    conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
    relu = nn.ReLU(inplace=True)
    conv2 = nn.Conv2d(hidden_channels, out_channels, 1)

    if fill_bias is not None:
        conv2.bias.data.fill_(fill_bias)

    output_head = nn.Sequential(conv1, relu, conv2)
    return output_head

class CenterNet(pl.LightningModule):
    """General CenterNet model. Build CenterNet from a given backbone and output
    """
    def __init__(self, backbone: Dict, num_classes: int, output_heads: Dict = None, optimizer: Dict = None, lr_scheduler: Dict = None, **kwargs):
        super(CenterNet, self).__init__()

        self.backbone = build_backbone(**backbone)        
        backbone_channels = self.backbone.out_channels
        self.output_stride = self.backbone.output_stride    # how much input image is downsampled
        
        self.num_classes = num_classes

        fill_bias = output_heads["fill_bias"]
        other_heads = output_heads["other_heads"]
        loss_weights = output_heads["loss_weights"]
        loss_functions = output_heads["loss_functions"]

        # create output heads and set their losses
        self.output_heads = nn.ModuleDict()
        self.head_loss_fn = nn.ModuleDict()
        self.output_heads["heatmap"] = _make_output_head(
            backbone_channels,
            backbone_channels,
            num_classes,
            fill_bias=fill_bias["heatmap"]
        )
        # self.head_loss_fn["heatmap"] = ModifiedFocalLossWithLogits(alpha=2., beta=4., reduction="sum")       # focal loss for heatmap
        self.head_loss_fn["heatmap"] = _supported_losses[loss_functions["heatmap"]](reduction="sum")
        assert "heatmap" in loss_weights

        for h in other_heads:
            assert h in _supported_heads
            assert h in loss_weights
            assert h in loss_functions
            self.output_heads[h] = _make_output_head(
                backbone_channels,
                backbone_channels,
                _output_head_channels[h],
                fill_bias=fill_bias[h]
            )
            # loss for size and offset head should be either L1Loss or SmoothL1Loss
            # cornernet uses smooth l1 loss, centernet uses l1 loss
            # NOTE: centernet author noted that l1 loss is better than smooth l1 loss
            self.head_loss_fn[h] = _supported_losses[loss_functions[h]](reduction="none")    # don't use reduction to apply mask later
        
        self.other_heads  = other_heads
        self.loss_weights = loss_weights   
        
        self.optimizer_cfg    = optimizer
        self.lr_scheduler_cfg = lr_scheduler

        self.save_hyperparameters()
        self._steps_per_epoch = None        # for steps_per_epoch property

    def forward(self, images):
        """Return encoded outputs, a dict of output feature maps. Use this output to either compute loss or decode to detections.
        """
        features = self.backbone(images)
        output = {}
        output["backbone_features"] = features      # for logging purpose
        
        for k, v in self.output_heads.items():
            output[k] = v(features)
        
        return output

    def compute_loss(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], eps: float = 1e-8):
        """Return a dict of losses for each output head, and weighted total loss. This method is called during the training step
        """
        bboxes = targets["bboxes"].clone()
        target_heatmap = targets["heatmap"]
        mask = targets["mask"].unsqueeze(-1)    # add column dimension to support broadcasting

        batch_size, _, heatmap_h, heatmap_w = preds["heatmap"].shape
        pred_heatmap = preds["heatmap"]
        size_map     = preds["size"].view(batch_size, 2, -1)    # flatten last xy dimensions for torch.gather() later
        offset_map   = preds["offset"].view(batch_size, 2, -1)

        # initialize losses
        losses = {}

        # convert normalized coordinates [0,1] to heatmap pixel coordinates [0,128]
        bboxes[...,[0,2]] *= heatmap_w
        bboxes[...,[1,3]] *= heatmap_h
        centers_int = bboxes[...,:2].long()     # convert to integer to use as index

        # combine xy indices for torch.gather()
        # repeat indices using .expand() to gather on 2 channels
        xy_indices = centers_int[...,1] * heatmap_w + centers_int[...,0]        # y * w + x
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size,2,-1))

        pred_sizes  = torch.gather(size_map, dim=-1, index=xy_indices)      # N2D
        pred_offset = torch.gather(offset_map, dim=-1, index=xy_indices)    # N2D

        # need to swapaxes since pred_size is N2D but true_wh is ND2
        target_size = bboxes[...,2:] * self.output_stride           # convert to original image pixel coordinate
        size_loss = self.head_loss_fn["size"](pred_sizes.swapaxes(1,2), target_size)
        size_loss = torch.sum(size_loss * mask)
        losses["size"] = size_loss

        target_offset = bboxes[...,:2] - torch.floor(bboxes[...,:2])    # quantization error
        offset_loss = self.head_loss_fn["offset"](pred_offset.swapaxes(1,2), target_offset)
        offset_loss = torch.sum(offset_loss * mask)
        losses["offset"] = offset_loss

        losses["heatmap"] = self.head_loss_fn["heatmap"](pred_heatmap, target_heatmap)

        # average over number of detections
        N = torch.sum(mask) + eps
        losses["size"] /= N
        losses["offset"] /= N
        losses["heatmap"] /= N

        total_loss = torch.tensor(0., dtype=losses["heatmap"].dtype, device=self.device)
        for k,v in losses.items():
            total_loss += v * self.loss_weights[k]
        losses["total"] = total_loss

        return losses

    def decode_detections(self, encoded_outputs: Dict[str, torch.Tensor], num_detections: int = 100, nms_kernel: int = 3, normalize_bbox: bool = False):
        """Decode model output to detections

        Args
            encoded_outputs: outputs after calling forward pass `model(images)`
            num_detections: number of detections to return. Default is 100
            nms_kernel: the kernel used for max pooling (pseudo-nms). Larger values will reduce false positives. Default is 3 (original paper)
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
        """
        # reference implementations
        # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L234
        # https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L118
        # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_decode.py#L28
        batch_size, _, out_h, out_w = encoded_outputs["heatmap"].shape
        heatmap    = encoded_outputs["heatmap"]
        size_map   = encoded_outputs["size"].view(batch_size, 2, -1)        # NCHW to NC(HW)
        offset_map = encoded_outputs["offset"].view(batch_size, 2, -1)

        # pseudo-nms via max pool
        # NOTE: must apply sigmoid before max pool
        heatmap = torch.sigmoid(heatmap)
        local_peaks = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=(nms_kernel-1)//2)
        nms_mask = (heatmap == local_peaks)
        heatmap = nms_mask.float() * heatmap

        # because there is only 1 size regression for each location,
        # there can't be multiple objects at the same heatmap location
        # thus we only consider the best candidate at each heatmap location
        heatmap, labels = torch.max(heatmap, dim=1)     # NHW

        # flatten to run topk
        heatmap = heatmap.view(batch_size, -1)          # N(HW)
        labels = labels.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(heatmap, num_detections)
        
        topk_labels = torch.gather(labels, dim=-1, index=topk_indices)

        # extract bboxes at topk positions
        # x, y are in output heatmap coordinates (128x128)
        # w, h are in input image coordinates (512x512)
        topk_x = topk_indices % out_w + torch.gather(offset_map[:,0], dim=-1, index=topk_indices)
        topk_y = topk_indices // out_w + torch.gather(offset_map[:,1], dim=-1, index=topk_indices)
        topk_w = torch.gather(size_map[:,0], dim=-1, index=topk_indices)
        topk_h = torch.gather(size_map[:,1], dim=-1, index=topk_indices)

        if normalize_bbox:
            # normalize to [0,1]
            topk_x /= out_w
            topk_y /= out_h
            topk_w /= (out_w * self.output_stride)
            topk_h /= (out_h * self.output_stride)
        else:
            # convert x, y to input image coordinates (512,512)
            topk_x *= self.output_stride
            topk_y *= self.output_stride

        topk_bboxes = torch.stack([topk_x, topk_y, topk_w, topk_h], dim=-1)  # NK4
        out = {
            "bboxes": topk_bboxes,
            "labels": topk_labels,
            "scores": topk_scores
        }
        return out

    # lightning method, return total loss here
    def training_step(self, batch, batch_idx):
        encoded_outputs = self(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch)
        for k,v in losses.items():
            self.log(f"train/{k}_loss", v)

        self.log("epoch_frac", self.global_step / self.steps_per_epoch)     # log this to view graph with epoch as x-axis

        self.log_histogram("output_values/heatmap", encoded_outputs["heatmap"])
        self.log_histogram("output_values/size", encoded_outputs["size"])
        self.log_histogram("output_values/offset", encoded_outputs["offset"])

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        encoded_outputs = self(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch)
        for k,v in losses.items():
            self.log(f"val/{k}_loss", v)

    def test_step(self, batch, batch_idx):
        encoded_outputs = self(batch["image"])
        detections = self.decode_detections(encoded_outputs)

    @torch.no_grad()
    def inference(self, data_dir, img_names, batch_size=4, num_detections=100, nms_kernel=3, save_path=None, score_threshold=0):
        dataset = InferenceDataset(data_dir, img_names)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_detections = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        self.eval()
        for batch in dataloader:
            img_widths = batch["original_width"].clone().numpy().reshape(-1,1,1)
            img_heights = batch["original_height"].clone().numpy().reshape(-1,1,1)

            encoded_outputs = self(batch["image"].to(self.device))
            detections = self.decode_detections(encoded_outputs, num_detections=num_detections, nms_kernel=nms_kernel, normalize_bbox=True)
            detections = {k: v.cpu().float().numpy() for k,v in detections.items()}

            detections["bboxes"][...,[0,2]] *= img_widths
            detections["bboxes"][...,[1,3]] *= img_heights

            for k, v in detections.items():
                all_detections[k].append(v)

        all_detections = {k: np.concatenate(v, axis=0) for k,v in all_detections.items()}
        
        if save_path is not None:
            bboxes = convert_cxcywh_to_xywh(detections["bboxes"]).tolist()
            labels = detections["labels"].tolist()
            scores = detections["scores"].tolist()

            detections_to_coco_results(range(len(img_names)), bboxes, labels, scores, save_path, score_threshold=score_threshold)

        return all_detections

    @property
    def steps_per_epoch(self):
        # does not consider multi-gpu training
        if self.trainer.max_steps:
            return self.trainer.max_steps
        
        if self._steps_per_epoch is None:
            self._steps_per_epoch = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        
        return self._steps_per_epoch

    def log_histogram(self, name: str, values: torch.Tensor):
        flatten_values = values.detach().view(-1).cpu().float().numpy()

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_histogram(name, flatten_values, global_step=self.global_step)
        
        elif isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({name: wandb.Histogram(flatten_values), "global_step": self.global_step})

    def register_optimizer(self, optimizer_cfg: Dict):
        self.optimizer_cfg = optimizer_cfg

    def register_lr_scheduler(self, lr_scheduler_cfg: Dict):
        self.lr_scheduler_cfg = lr_scheduler_cfg

    # lightning method
    def configure_optimizers(self):
        if self.optimizer_cfg is None:
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

        # override default behavior to update lr every step instead of epoch
        return_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
        return return_dict

class CenterNetDetectionTorchScript(nn.Module):
    def __init__(self, backbone: Dict, num_classes: int, **kwargs):
        super().__init__()
        self.backbone = build_backbone(**backbone)        
        backbone_channels = self.backbone.out_channels

        self.output_heatmap = _make_output_head(backbone_channels, backbone_channels, num_classes)
        self.output_size = _make_output_head(backbone_channels, backbone_channels, 2)
        self.output_offset = _make_output_head(backbone_channels, backbone_channels, 2)
    
    @classmethod
    def load_centernet_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]
        state_dict = checkpoint["state_dict"]

        model = cls(**hparams)

        output_weights = [x for x in state_dict.keys() if x.startswith("output_heads.")]
        remove_length = len("output_heads.")

        for old_key in output_weights:
            new_key = "output_" + old_key[remove_length:]
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]

        model.load_state_dict(state_dict)
        return model

    def forward(self, images):
        features = self.backbone(images)

        heatmap = self.output_heatmap(features)
        heatmap = torch.sigmoid(heatmap)
        size = self.output_size(features)
        offset = self.output_offset(features)

        return heatmap, size, offset
