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
from tqdm import tqdm
import cv2

try:
    import wandb
except ImportError:
    pass

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_output_heads, HeatmapHead, Box2DHead, ReIDHead
from .tracker import Tracker
from ..datasets import InferenceDataset
from ..utils import load_config, draw_bboxes, convert_cxcywh_to_xywh, convert_x1y1x2y2_to_xywh
from ..eval import detections_to_coco_results, evaluate_coco_detection, evaluate_mot_tracking_sequence

_output_templates = {
    "detection": namedtuple("Detection", "heatmap box_2d"),
    "tracking": namedtuple("Tracking", "heatmap box_2d reid")
}

class CenterNet(pl.LightningModule):
    """General CenterNet model
    """
    def __init__(self, backbone: Dict, neck: Dict, output_heads: Dict, task: str, optimizer: Dict = None, lr_scheduler: Dict = None, **kwargs):
        """Build CenterNet from backbone, neck, and output heads configurations
        """
        super().__init__()

        return_features = True if neck["name"] in ("fpn", "bifpn", "ida") else False
        self.backbone = build_backbone(backbone, return_features=return_features)
        self.neck = build_neck(neck, backbone_channels=self.backbone.out_channels)
        self.output_heads = build_output_heads(output_heads, in_channels=self.neck.out_channels)

        self.output_stride = self.backbone.output_stride // self.neck.upsample_stride
        
        self.task = task
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.num_classes = output_heads["heatmap"]["num_classes"]

        self.save_hyperparameters()
        self._steps_per_epoch = None
        self.example_input_array = torch.rand((1,3,512,512))    # for model logging

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

    def on_validation_epoch_start(self):
        if self.task == "tracking":
            self.tracker = Tracker(device=self.device)

    def validation_step(self, batch, batch_idx): 
        encoded_outputs = self.get_encoded_outputs(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch, ignore_reid=True)    # during validation, only evaluate detection loss
        for k,v in losses.items():
            self.log(f"val/{k}_loss", v)
        
        # gather detections for evaluation
        # use torchmetrics to handle this instead?
        # https://github.com/PyTorchLightning/metrics
        # or a plain Metric class to handle gathering and reducing logic?
        mask = batch["mask"].cpu().numpy().astype(bool)
        bboxes = batch["bboxes"].cpu().numpy()

        if self.task == "detection":
            bboxes = convert_cxcywh_to_xywh(bboxes)
            labels = batch["labels"].cpu().numpy()
            target = {
                "bboxes": [box[m] for box, m in zip(bboxes, mask)],    # list of 1-d np.ndarray of different lengths
                "labels": [label[m] for label, m in zip(labels, mask)]
            }
            
            preds = self.gather_detection2d(encoded_outputs["heatmap"].sigmoid(), encoded_outputs["box_2d"], normalize_bbox=True)
            preds = {k: v.cpu().numpy() for k,v in preds.items()}           # 2-d np array with dim batch_size x num_detections (100)
            preds["bboxes"] = convert_x1y1x2y2_to_xywh(preds["bboxes"])
        
        elif self.task == "tracking":
            bboxes = convert_cxcywh_to_xywh(bboxes)
            track_ids = batch["ids"].cpu().numpy()
            target = {
                "bboxes": [box[m] for box, m in zip(bboxes, mask)],
                "track_ids": [track_id[m] for track_id, m in zip(track_ids, mask)]
            }
            
            detections = self.gather_tracking2d(encoded_outputs["heatmap"].sigmoid(), encoded_outputs["box_2d"], encoded_outputs["reid"], normalize_bbox=True)
            detections = {k: v.cpu().numpy() for k,v in detections.items()}
            pred_bboxes = []
            pred_track_ids = []

            # use Tracker.update() instead of Tracker.step_batch() to avoid running forward pass twice
            for b in range(detections["bboxes"].shape[0]):
                new_bboxes = detections["bboxes"][b]
                new_labels = detections["labels"][b]
                new_scores = detections["scores"][b]
                new_embeddings = detections["embeddings"][b]
                self.tracker.update(new_bboxes, new_labels, new_scores, new_embeddings)
                
                track_bboxes = [convert_x1y1x2y2_to_xywh(x.bbox) for x in self.tracker.tracks if x.active]
                track_ids = [x.track_id for x in self.tracker.tracks if x.active]
                pred_bboxes.append(track_bboxes)
                pred_track_ids.append(track_ids)
            
            preds = {"bboxes": pred_bboxes, "track_ids": pred_track_ids}
        
        return preds, target

    def validation_epoch_end(self, outputs):
        preds = {key: [] for key in outputs[0][0]}
        target = {key: [] for key in outputs[0][1]}

        # concatenate lists
        for (pred_detections, target_detections) in outputs:
            for key in preds:
                preds[key].extend(pred_detections[key])
            for key in target:
                target[key].extend(target_detections[key])
        
        if self.task == "detection":
            metrics = evaluate_coco_detection(
                preds["bboxes"], preds["labels"], preds["scores"], 
                target["bboxes"], target["labels"],
                metrics_to_return=("AP", "AP50", "AP75")
            )
        elif self.task == "tracking":
            metrics = evaluate_mot_tracking_sequence(preds["bboxes"], preds["track_ids"], target["bboxes"], target["track_ids"])
            self.tracker = None
        
        for metric, value in metrics.items():
            self.log(f"val/{metric}", value)

    # this is needed for some learning rate schedulers e.g. OneCycleLR
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-774265729
    def get_steps_per_epoch(self):
        if self._steps_per_epoch is None:
            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
            if self.trainer.tpu_cores:
                num_devices = max(num_devices, self.trainer.tpu_cores)
            
            self._steps_per_epoch = len(self.train_dataloader()) // (num_devices * self.trainer.accumulate_grad_batches)
        
        return self._steps_per_epoch

    def log_histogram(self, name: str, values: torch.Tensor, freq=500):
        """Log histogram. Only TensorBoard and Wandb are supported
        """
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

    def configure_optimizers(self):
        if self.optimizer_cfg is None:
            warnings.warn("Optimizer config was not specified. Using AdamW optimizer with lr=5e-4")
            optimizer_params = dict(lr=5e-4)
            self.optimizer_cfg = dict(name="AdamW", params=optimizer_params)

        optimizer_class = torch.optim.__dict__[self.optimizer_cfg["name"]]
        optimizer = optimizer_class(self.parameters(), **self.optimizer_cfg["params"])

        if self.lr_scheduler_cfg is None:
            return optimizer
            
        scheduler_class = torch.optim.lr_scheduler.__dict__[self.lr_scheduler_cfg["name"]]
        if self.lr_scheduler_cfg["name"] == "OneCycleLR":
            self.lr_scheduler_cfg["params"]["max_lr"] = self.optimizer_cfg["params"]["lr"]
            lr_scheduler = scheduler_class(optimizer, epochs=self.trainer.max_epochs, steps_per_epoch=self.get_steps_per_epoch(), **self.lr_scheduler_cfg["params"])

            # OneCycleLR should be called every train step (per batch)
            return_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }

        else:
            lr_scheduler = scheduler_class(optimizer, **self.lr_scheduler_cfg["params"])
            return_dict = {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler
            }

        return return_dict

    def gather_detection2d(self, heatmap: torch.Tensor, box_2d: torch.Tensor, num_detections: int = 100, nms_kernel: int = 3, normalize_bbox: bool = False):
        """Decode model outputs for detection task

        Args
            heatmap: heatmap output
            box_2d: box_2d output
            num_detections: number of detections to return. Default is 100
            nms_kernel: the kernel used for max pooling (pseudo-nms). Larger values will reduce false positives. Default is 3 (original paper)
            normalize_bbox: whether to normalize bbox coordinates to [0,1]. Otherwise bbox coordinates are in input image coordinates. Default is False
        """
        topk_scores, topk_indices, topk_labels = HeatmapHead.gather_topk(heatmap, nms_kernel=nms_kernel, num_detections=num_detections)
        topk_bboxes = Box2DHead.gather_at_indices(box_2d, topk_indices, normalize_bbox=normalize_bbox, stride=self.output_stride)

        out = {
            "bboxes": topk_bboxes,
            "labels": topk_labels,
            "scores": topk_scores
        }
        return out

    def gather_tracking2d(self, heatmap: torch.Tensor, box_2d: torch.Tensor, reid: torch.Tensor, num_detections=100, nms_kernel=3, normalize_bbox=False):
        """Decode model outputs for tracking task
        """
        topk_scores, topk_indices, topk_labels = HeatmapHead.gather_topk(heatmap, nms_kernel=nms_kernel, num_detections=num_detections)
        topk_bboxes = Box2DHead.gather_at_indices(box_2d, topk_indices, normalize_bbox=normalize_bbox, stride=self.output_stride)
        topk_embeddings = ReIDHead.gather_at_indices(reid, topk_indices)

        out = {
            "bboxes": topk_bboxes,
            "labels": topk_labels,
            "scores": topk_scores,
            "embeddings": topk_embeddings
        }
        return out

    @torch.no_grad()
    def inference_detection2d(self, data_dir, img_names, batch_size=4, num_detections=100, nms_kernel=3, save_path=None, score_threshold=0):
        """Run detection on a folder of images
        """
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
            detections = self.gather_detection2d(heatmap, box_2d, num_detections=num_detections, nms_kernel=nms_kernel, normalize_bbox=True)
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
    def inference_tracking2d(self, data_dir, batch_size=4, save_dir=None, save_results=False, save_images=False, **kwargs):
        """Run tracking on a folder of images
        """
        tracker = Tracker(self, **kwargs)

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

                            # MOT challenge format uses 1-based indexing
                            line = f"{frame+i+1},{track_id+1},{x1+1},{y1+1},{x2-x1},{y2-y1},-1,-1,-1,-1\n"
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
    # there will be signature change: https://github.com/PyTorchLightning/pytorch-lightning/pull/8697
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
