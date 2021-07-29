from typing import Dict
import warnings
from collections import namedtuple
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb
from tqdm import tqdm
import cv2

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_output_heads
from .tracker import MatchingCost, Tracker
from ..datasets import InferenceDataset
from ..utils import load_config, draw_bboxes
from ..eval import detections_to_coco_results

_output_templates = {
    "detection": namedtuple("Detection", "heatmap box_2d"),
    "tracking": namedtuple("Tracking", "heatmap box_2d reid")
}

class CenterNet(pl.LightningModule):
    """General CenterNet model. Build CenterNet from a given backbone and output
    """
    def __init__(self, backbone: Dict, neck: Dict, output_heads: Dict, task: str, optimizer: Dict = None, lr_scheduler: Dict = None, **kwargs):
        super().__init__()

        return_features = True if neck["name"] in ("fpn") else False
        self.backbone = build_backbone(backbone, return_features=return_features)
        self.neck = build_neck(neck, backbone_channels=self.backbone.out_channels)
        self.output_heads = build_output_heads(output_heads, in_channels=self.neck.out_channels)

        self.output_stride = self.backbone.output_stride // self.neck.upsample_stride
        
        self.task = task
        self.optimizer_cfg    = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.num_classes = output_heads["heatmap"]["num_classes"]

        self.save_hyperparameters()
        self._steps_per_epoch = None

    def forward(self, x):
        """Return encoded outputs. Use namedtuple to support TorchScript and ONNX export. Heatmap is after sigmoid
        """
        encoded_outputs = self.get_encoded_outputs(x)
        encoded_outputs["heatmap"] = torch.sigmoid(encoded_outputs["heatmap"])

        # create a namedtuple
        template = _output_templates[self.task]
        outputs = {name: encoded_outputs[name] for name in template._fields}
        outputs = template(**outputs)

        return outputs

    def get_encoded_outputs(self, x):
        """Return encoded outputs, a dict of output feature maps. Use this output to either compute loss or decode to detections. Heatmap is before sigmoid
        """
        features = self.backbone(x)
        features = self.neck(features)
        output = {}
        output["features"] = features      # for logging purpose
        
        for k, v in self.output_heads.items():
            output[k] = v(features)
        
        return output

    def compute_loss(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], ignore_reid=False, eps: float = 1e-8):
        """Return a dict of losses for each output head, and weighted total loss. This method is called during the training step
        """
        losses = {"total": torch.tensor(0., device=self.device)}
        for name, head in self.output_heads.items():
            if ignore_reid and name == "reid":
                continue
            losses[name] = head.compute_loss(preds, targets, eps=eps)
            losses["total"] += losses[name] * head.loss_weight

        return losses

    # lightning method, return total loss here
    def training_step(self, batch, batch_idx):
        encoded_outputs = self.get_encoded_outputs(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch)
        for k,v in losses.items():
            self.log(f"train/{k}_loss", v)

        for k,v in encoded_outputs.items():
            self.log_histogram(f"output_values/{k}", v)

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        encoded_outputs = self.get_encoded_outputs(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch, ignore_reid=True)    # during validation, only evaluate detection loss
        for k,v in losses.items():
            self.log(f"val/{k}_loss", v)
        # TODO: evaluation

    def get_steps_per_epoch(self):
        # does not consider multi-gpu training
        if self.trainer.max_steps:
            return self.trainer.max_steps
        
        if self._steps_per_epoch is None:
            self._steps_per_epoch = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        
        return self._steps_per_epoch

    def log_histogram(self, name: str, values: torch.Tensor, freq=500):
        if self.trainer.global_step % freq != 0:
            return

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

    def decode_heatmap(self, heatmap: torch.Tensor, nms_kernel=3, num_detections=100):
        batch_size = heatmap.shape[0]

        # pseudo-nms via max pool
        padding = (nms_kernel - 1) // 2
        nms_mask = F.max_pool2d(heatmap, kernel_size=nms_kernel, stride=1, padding=padding) == heatmap
        heatmap *= nms_mask
        
        # since box regression is shared, we only consider the best candidate at each heatmap location
        heatmap, labels = torch.max(heatmap, dim=1)

        # flatten to run topk
        heatmap = heatmap.view(batch_size, -1)
        labels = labels.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(heatmap, num_detections)
        topk_labels = torch.gather(labels, dim=-1, index=topk_indices)

        return topk_scores, topk_indices, topk_labels

    def decode_box2d(self, box_2d: torch.Tensor, indices, normalize_bbox=False):
        batch_size, _, output_height, output_width = box_2d.shape

        cx = indices % output_width + 0.5
        cy = indices // output_width + 0.5

        box_2d = box_2d.view(batch_size, 4, -1)
        x1 = cx - torch.gather(box_2d[:,0], dim=-1, index=indices)    # x1 = cx - left
        y1 = cy - torch.gather(box_2d[:,1], dim=-1, index=indices)    # y1 = cy - top
        x2 = cx + torch.gather(box_2d[:,2], dim=-1, index=indices)    # x2 = cx + right
        y2 = cy + torch.gather(box_2d[:,3], dim=-1, index=indices)    # y2 = cy + bottom

        topk_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        if normalize_bbox:
            # normalize to [0,1]
            topk_bboxes[...,[0,2]] /= output_width
            topk_bboxes[...,[1,3]] /= output_height    
        else:
            # convert to input image coordinates
            topk_bboxes *= self.output_stride
        
        return topk_bboxes

    def decode_reid(self, reid: torch.Tensor, indices: torch.Tensor):
        batch_size, embedding_size, _, _ = reid.shape

        reid = reid.view(batch_size, embedding_size, -1)
        indices = indices.unsqueeze(1).expand(batch_size, embedding_size, -1)
        embeddings = torch.gather(reid, dim=-1, index=indices)
        embeddings = embeddings.swapaxes(1,2)
        return embeddings

    def decode_detection(self, heatmap: torch.Tensor, box_2d: torch.Tensor, num_detections: int = 100, nms_kernel: int = 3, normalize_bbox: bool = False):
        """Decode model outputs for detection task

        Args
            heatmap: heatmap output
            box_2d: box_2d output
            num_detections: number of detections to return. Default is 100
            nms_kernel: the kernel used for max pooling (pseudo-nms). Larger values will reduce false positives. Default is 3 (original paper)
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
        """
        # reference implementations
        # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L234
        # https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L118
        # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_decode.py#L28
        topk_scores, topk_indices, topk_labels = self.decode_heatmap(heatmap, nms_kernel=nms_kernel, num_detections=num_detections)
        topk_bboxes = self.decode_box2d(box_2d, topk_indices, normalize_bbox=normalize_bbox)

        out = {
            "bboxes": topk_bboxes,
            "labels": topk_labels,
            "scores": topk_scores
        }
        return out

    def decode_tracking(self, heatmap: torch.Tensor, box_2d: torch.Tensor, reid: torch.Tensor, num_detections=100, nms_kernel=3, normalize_bbox=False):
        """Decode model outputs for tracking task
        """
        topk_scores, topk_indices, topk_labels = self.decode_heatmap(heatmap, nms_kernel=nms_kernel, num_detections=num_detections)
        topk_bboxes = self.decode_box2d(box_2d, topk_indices, normalize_bbox=normalize_bbox)
        topk_embeddings = self.decode_reid(reid, topk_indices)

        out = {
            "bboxes": topk_bboxes,
            "labels": topk_labels,
            "scores": topk_scores,
            "embeddings": topk_embeddings
        }
        return out

    @torch.no_grad()
    def inference_detection(self, data_dir, img_names, batch_size=4, num_detections=100, nms_kernel=3, save_path=None, score_threshold=0):
        transforms = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2()
        ])
        dataset = InferenceDataset(data_dir, img_names, transforms=transforms, file_ext=".jpg")
        dataloader = DataLoader(dataset, batch_size=batch_size)

        all_detections = {
            "bboxes": [],
            "labels": [],
            "scores": []
        }

        self.eval()
        for batch in tqdm(dataloader):
            img_widths = batch["original_width"].clone().numpy().reshape(-1,1,1)
            img_heights = batch["original_height"].clone().numpy().reshape(-1,1,1)

            heatmap, box_2d = self(batch["image"].to(self.device))
            detections = self.decode_detection(heatmap, box_2d, num_detections=num_detections, nms_kernel=nms_kernel, normalize_bbox=True)
            detections = {k: v.cpu().float().numpy() for k,v in detections.items()}

            detections["bboxes"][...,[0,2]] *= img_widths
            detections["bboxes"][...,[1,3]] *= img_heights

            for k, v in detections.items():
                all_detections[k].append(v)

        all_detections = {k: np.concatenate(v, axis=0) for k,v in all_detections.items()}
        
        if save_path is not None:
            bboxes = detections["bboxes"].tolist()
            labels = detections["labels"].tolist()
            scores = detections["scores"].tolist()

            detections_to_coco_results(range(len(img_names)), bboxes, labels, scores, save_path, score_threshold=score_threshold)

        return all_detections

    @torch.no_grad()
    def inference_tracking(self, data_dir, batch_size=4, save_dir=None, save_results=False, save_images=False):
        matching_cost = MatchingCost(reid_weight=1, box_weight=0)
        tracker = Tracker(self, device=self.device, detection_threshold=0.3, num_detections=300, matching_cost=matching_cost, smoothing_factor=0.5)

        transforms = A.Compose([
            A.Resize(height=608, width=1088),
            A.Normalize(),
            ToTensorV2(),
        ])
        dataset = InferenceDataset(data_dir, transforms=transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            results_path = os.path.join(save_dir, "tracking_results.txt")
            images_dir = os.path.join(save_dir, "images")
            if save_results:
                if os.path.exists(results_path):
                    os.remove(results_path)
            if save_images:
                os.makedirs(images_dir, exist_ok=True)

        elif save_results or save_images:
            warnings.warn("save_dir is not specified. results and images won't be saved")
            save_results = False
            save_images = False

        self.eval()
        frame = 0
        for batch in tqdm(dataloader):
            img_paths = batch["image_path"]
            img_widths = batch["original_width"].clone().numpy()
            img_heights = batch["original_height"].clone().numpy()
            
            out = tracker.step_batch(batch["image"])
            track_bboxes = out["bboxes"]
            track_ids = out["track_ids"]

            # write tracking results to file
            if save_results:
                with open(os.path.join(save_dir, "tracking_results.txt"), "a") as f:
                    for i, (frame_bboxes, frame_track_ids, img_w, img_h) in enumerate(zip(track_bboxes, track_ids, img_widths, img_heights)):
                        for box, track_id in zip(frame_bboxes, frame_track_ids):
                            x1 = box[0] * img_w
                            y1 = box[1] * img_h
                            x2 = box[2] * img_w
                            y2 = box[3] * img_h

                            line = f"{frame+i+1}, {track_id}, {x1+1}, {y1+1}, {x2-x1}, {y2-y1}, -1, -1, -1, -1\n"
                            f.write(line)
                            
            if save_images:
                for i, (frame_bboxes, frame_track_ids, img_p) in enumerate(zip(track_bboxes, track_ids, img_paths)):
                    img = cv2.imread(img_p)
                    draw_bboxes(img, frame_bboxes, frame_track_ids, normalized_bbox=True, text_color=(255,255,255))
                    
                    save_img_path = os.path.join(images_dir, f"{frame+i}.jpg")
                    cv2.imwrite(save_img_path, img)

            frame += len(track_ids)

    # allow loading checkpoint with mismatch weights
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/4690#issuecomment-731152036
    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    warnings.warn(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

def build_centernet(config):
    if isinstance(config, str):
        config = load_config(config)
        config = config["model"]
    
    if "load_from_checkpoint" in config:
        model = CenterNet.load_from_checkpoint(config["load_from_checkpoint"], strict=False, **config)
    else:
        model = CenterNet(**config)
    return model
