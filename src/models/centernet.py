from typing import Dict
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_output_heads
from ..datasets import InferenceDataset
from ..utils import convert_cxcywh_to_xywh, load_config
from ..eval import detections_to_coco_results

class CenterNet(pl.LightningModule):
    """General CenterNet model. Build CenterNet from a given backbone and output
    """
    def __init__(self, backbone: Dict, neck: Dict, output_heads: Dict, task: str, optimizer: Dict = None, lr_scheduler: Dict = None, **kwargs):
        super().__init__()

        return_features = True if neck["name"] in ("fpn") else False
        self.backbone = build_backbone(backbone, return_features=return_features)
        self.neck = build_neck(neck, backbone_channels=self.backbone.out_channels)
        self.output_heads = build_output_heads(output_heads, in_channels=self.neck.out_channels)

        self.output_stride = self.backbone.output_stride    # how much input image is downsampled
        
        self.task = task
        self.optimizer_cfg    = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.num_classes = output_heads["heatmap"]["num_classes"]

        self.save_hyperparameters()
        self._steps_per_epoch = None

    def forward(self, x):
        """Return encoded outputs, a dict of output feature maps. Use this output to either compute loss or decode to detections.
        """
        features = self.backbone(x)
        features = self.neck(features)
        output = {}
        output["features"] = features      # for logging purpose
        
        for k, v in self.output_heads.items():
            output[k] = v(features)
        
        return output

    def compute_loss(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], eps: float = 1e-8):
        """Return a dict of losses for each output head, and weighted total loss. This method is called during the training step
        """
        losses = {"total": torch.tensor(0., device=self.device)}
        for name, head in self.output_heads.items():
            losses[name] = head.compute_loss(preds, targets, eps=eps)
            losses["total"] += losses[name] * head.loss_weight

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

        self.log("epoch_frac", self.global_step / self.get_steps_per_epoch())     # log this to view graph with epoch as x-axis

        self.log_histogram("output_values/heatmap", encoded_outputs["heatmap"])
        self.log_histogram("output_values/box_2d", encoded_outputs["box_2d"])

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

    def get_steps_per_epoch(self):
        # does not consider multi-gpu training
        if self.trainer.max_steps:
            return self.trainer.max_steps
        
        if self._steps_per_epoch is None:
            self._steps_per_epoch = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        
        return self._steps_per_epoch

    def log_histogram(self, name: str, values: torch.Tensor, freq=100):
        if self.trainer.global_step % freq == 0:
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
            lr_scheduler = lr_scheduler_algo(optimizer, epochs=self.trainer.max_epochs, steps_per_epoch=self.get_steps_per_epoch(), **self.lr_scheduler_cfg["params"])
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

def build_centernet(config):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]

    model = CenterNet(**config)
    return model


# class CenterNetDetection(nn.Module):
#     def __init__(self, backbone: Dict, num_classes: int, **kwargs):
#         super().__init__()
#         self.backbone = build_backbone(**backbone)
#         backbone_channels = self.backbone.out_channels

#         self.output_heatmap = _make_output_head(backbone_channels, backbone_channels, num_classes)
#         self.output_size = _make_output_head(backbone_channels, backbone_channels, 2)
#         self.output_offset = _make_output_head(backbone_channels, backbone_channels, 2)
    
#     @classmethod
#     def load_centernet_from_checkpoint(cls, checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         hparams = checkpoint["hyper_parameters"]
#         state_dict = checkpoint["state_dict"]

#         model = cls(**hparams)

#         output_weights = [x for x in state_dict.keys() if x.startswith("output_heads.")]
#         remove_length = len("output_heads.")

#         for old_key in output_weights:
#             new_key = "output_" + old_key[remove_length:]
#             state_dict[new_key] = state_dict[old_key]
#             del state_dict[old_key]

#         model.load_state_dict(state_dict)
#         return model

#     def forward(self, images):
#         features = self.backbone(images)

#         heatmap = self.output_heatmap(features)
#         heatmap = torch.sigmoid(heatmap)
#         size = self.output_size(features)
#         offset = self.output_offset(features)

#         return heatmap, size, offset
