from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

import math
import cv2
import os
import matplotlib.pyplot as plt

from losses import FocalLossWithLogits, render_target_heatmap
from metrics import eval_detections
from utils import convert_cxcywh_to_x1y1x2y2, draw_bboxes, draw_heatmap

_resnet_mapper = {
    "resnet18": torchvision.models.resnet.resnet18,
    "resnet34": torchvision.models.resnet.resnet34,
    "resnet50": torchvision.models.resnet.resnet50, 
    "resnet101": torchvision.models.resnet.resnet101
}

_optimizer_mapper = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop
}

RED = (1., 0., 0.)
BLUE = (0., 0., 1.)

class UpsampleBlock(nn.Module):
    """Upsample block (convolution transpose) with optional DCN (currently not supported)

    Architecture: conv + conv transpose, with BN and relu
    """
    # NOTE: architecture choices
    # conv + deconv (centernet)
    # deconv + conv
    # upsampling + conv
    def __init__(
        self, in_channels: int, out_channels: int, 
        deconv_kernel: int, deconv_stride: int=2,
        deconv_pad: int=1, deconv_out_pad: int=0, 
        dcn: bool=False, init_bilinear: bool=True):
        
        super(UpsampleBlock, self).__init__()
        if dcn:
            # potential dcn implementations
            # torchvision: https://pytorch.org/vision/stable/ops.html#torchvision.ops.deform_conv2d
            # detectron: https://detectron2.readthedocs.io/en/latest/modules/layers.html#detectron2.layers.ModulatedDeformConv
            # mmcv: https://mmcv.readthedocs.io/en/stable/api.html#mmcv.ops.DeformConv2d
            raise NotImplementedError()
        
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn_conv = nn.BatchNorm2d(out_channels)

        # NOTE: how to choose padding for conv transpose?
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, deconv_kernel, stride=deconv_stride,
            padding=deconv_pad, output_padding=deconv_out_pad, bias=False)
        self.bn_deconv = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # default behavior, initialize weights to bilinear upsampling
        # TF CenterNet does not do this
        if init_bilinear:
            self.init_bilinear_upsampling()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn_conv(out)
        out = self.relu(out)

        out = self.deconv(out)
        out = self.bn_deconv(out)
        out = self.relu(out)

        return out

    def init_bilinear_upsampling(self):
        # initialize convolution transpose layer as bilinear upsampling
        # https://github.com/ucbdrive/dla/blob/master/dla_up.py#L26-L33
        w = self.deconv.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2*f - 1 - f%2) / (f*2.)
        
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0,0,i,j] = (1 - math.fabs(i/f - c)) * (1 - math.fabs(j/f - c))
        
        for c in range(1, w.size(0)):
            w[c,0,:,:] = w[0,0,:,:]

class ResNetBackbone(nn.Module):
    """Modified PoseNet from CenterNet https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py

    Original PoseNet https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
    
    TF implementations: https://github.com/tensorflow/models/blob/master/research/object_detection/models/center_net_resnet_feature_extractor.py
    """
    def __init__(self, model: str="resnet50", pretrained: bool=True, upsample_init_bilinear: bool=True):
        super(ResNetBackbone, self).__init__()
        # downsampling path from resnet
        backbone = _resnet_mapper[model](pretrained=pretrained)
        self.downsample = nn.Sequential(
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        resnet_out_channels = 2048
        
        # upsampling parameters
        # NOTE: should these be included in constructor arguments?
        up_channels = [256, 128, 64]    # from CenterNet paper. original PoseNet uses [256,256,256]
        up_kernels = [4, 4, 4]          # from PoseNet paper
        self.upsample = self._make_upsample_stage(
            in_channels=resnet_out_channels, 
            up_channels=up_channels, 
            up_kernels=up_kernels,
            init_bilinear=upsample_init_bilinear)
        self.out_channels = up_channels[-1]

    def forward(self, x):
        out = self.downsample(x)
        out = self.upsample(out)

        return out

    def _make_upsample_stage(
        self,
        in_channels: int, 
        up_channels: List[int],
        up_kernels: List[int],
        init_bilinear: bool=True
        ):
        layers = []
        layers.append(UpsampleBlock(
            in_channels, up_channels[0], deconv_kernel=up_kernels[0], 
            init_bilinear=init_bilinear
        ))

        for i in range(len(up_channels)-1):
            layers.append(UpsampleBlock(
                up_channels[i], up_channels[i+1], deconv_kernel=up_kernels[i+1], 
                init_bilinear=init_bilinear
            ))

        return nn.Sequential(*layers)

class OutputHead(nn.Module):
    """ Output head for CenterNet. Reference implementation https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_head.py
    """
    def __init__(
        self, in_channels: int, out_channels: int, 
        fill_bias: float=None
        ):
        super(OutputHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

        if fill_bias != None:
            self.conv2.bias.data.fill_(fill_bias)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        return out

class CenterNet(pl.LightningModule):
    """General CenterNet model. Build CenterNet from a given backbone and output
    """
    supported_heads = ["size", "offset"]
    output_head_channels ={
        "size": 2,
        "offset": 2
    }

    def __init__(
        self, 
        backbone: nn.Module, 
        num_classes: int, 
        other_heads: Iterable[str]=["size", "offset"], 
        loss_weights: Dict[str,float]=dict(size=0.1,offset=1),
        heatmap_bias: float=None,
        max_pool_kernel: int=3,
        num_detections: int=40,
        batch_size: int=4,
        optimizer: str="adam",
        lr: float=1e-3
        ):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.out_channels
        self.num_classes = num_classes
        self.optimizer_name = optimizer

        # for heatmap output, fill a pre-defined bias value
        # for other outputs, fill bias with 0 to match identity mapping (from centernet)
        self.output_heads = nn.ModuleDict()
        self.output_heads["heatmap"] = OutputHead(
            feature_channels, num_classes, 
            fill_bias=heatmap_bias)
        # other_heads excludes the compulsory heatmap head
        for h in other_heads:
            assert h in self.supported_heads
            self.output_heads[h] = OutputHead(
                feature_channels, self.output_head_channels[h], 
                fill_bias=0)
        self.other_heads = other_heads

        # loss weights are used to calculated total weighted loss
        for x in other_heads:
            assert x in loss_weights
        self.loss_weights = loss_weights   
        
        # parameterized focal loss for heatmap
        self.focal_loss = FocalLossWithLogits(alpha=2., beta=4.)

        # for detection decoding
        # this is used to mimic nms
        self.nms_max_pool = nn.MaxPool2d(max_pool_kernel, stride=1, padding=(max_pool_kernel-1)//2)     # same padding
        self.num_detections = num_detections

        # for pytorch lightning tuner
        self.batch_size = batch_size
        self.learning_rate = lr

        # log hyperparameters
        self.save_hyperparameters({"backbone": backbone.__class__.__name__})
        self.save_hyperparameters("num_classes", "other_heads", "loss_weights", "heatmap_bias", "max_pool_kernel", "num_detections", "batch_size", "optimizer", "lr")      

    def forward(self, batch):
        """Return a dictionary of feature maps for each output head. Use this output to either decode to predictions or compute loss.
        """
        img = batch["image"]

        features = self.backbone(img)
        output = {}
        output["backbone_features"] = features
        for k,v in self.output_heads.items():
            # k is head name, v is that head nn.Module
            output[k] = v(features)
        
        return output

    def compute_loss(self, output_maps: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Return a dictionary of losses for each output head. This method is called during the training step
        """
        bboxes = targets["bboxes"].clone()               # ND4
        labels = targets["labels"]               # ND
        mask = targets["mask"].unsqueeze(-1)     # add column dimension to support broadcasting

        heatmap = output_maps["heatmap"]
        batch_size, channels, output_h, output_w = heatmap.shape

        size_map = output_maps["size"].view(batch_size, 2, -1)      # flatten last xy dimensions
        offset_map = output_maps["offset"].view(batch_size, 2, -1)  # for torch.gather() later

        # initialize losses to 0
        losses = {
            "heatmap": torch.tensor(0., dtype=torch.float32, device=self.device)
        }

        centers = bboxes[:,:,:2].clone()    # x and y, relative scale
        true_wh = bboxes[:,:,2:]            # w and h, relative scale

        centers[:,:,0] *= output_w      # convert to absolute scale
        centers[:,:,1] *= output_h
        centers_int = centers.long()    # convert to integer to use as index

        # combine xy indices for torch.gather()
        # repeat indices using .expand() to gather on 2 channels
        xy_indices = centers_int[:,:,1] * output_w + centers_int[:,:,0]     # y * w + x
        xy_indices = xy_indices.unsqueeze(1).expand((batch_size,2,-1))

        pred_sizes = torch.gather(size_map, dim=-1, index=xy_indices)       # N2D
        pred_offset = torch.gather(offset_map, dim=-1, index=xy_indices)    # N2D

        # need to swapaxes since pred_size is N2D but true_wh is ND2
        # use the mask to ignore none detections due to padding
        # NOTE: author noted that l1 loss is better than smooth l1 loss
        size_loss = F.l1_loss(pred_sizes.swapaxes(1,2), true_wh, reduction="none")
        size_loss = torch.sum(size_loss * mask)
        losses["size"] = size_loss

        # NOTE: offset is in absolute scale (float number) of output heatmap
        offset_loss = F.l1_loss(pred_offset.swapaxes(1,2), centers - torch.floor(centers), reduction="none")
        offset_loss = torch.sum(offset_loss * mask)
        losses["offset"] = offset_loss

        for b in range(batch_size):
            # convert wh to absolute scale
            abs_wh = true_wh[b].clone()
            abs_wh[:,0] *= output_w
            abs_wh[:,1] *= output_h

            # render target heatmap and accumulate focal loss
            target_heatmap = render_target_heatmap(
                heatmap.shape[1:], centers_int[b], abs_wh, 
                labels[b], mask[b], device=self.device)
            losses["heatmap"] += self.focal_loss(heatmap[b], target_heatmap)

        # average over number of detections
        N = torch.sum(mask)
        losses["heatmap"] /= N
        losses["size"] /= N
        losses["offset"] /= N

        return losses

    def decode_detections(self, encoded_output: Dict[str, torch.Tensor], num_detections: int=None):
        """Decode model output to detections
        """
        # reference implementations
        # https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py#L234
        # https://github.com/developer0hye/Simple-CenterNet/blob/main/models/centernet.py#L118
        # https://github.com/lbin/CenterNet-better-plus/blob/master/centernet/centernet_decode.py#L28
        if num_detections == None:
            num_detections = self.num_detections

        batch_size, channels, height, width = encoded_output["heatmap"].shape
        heatmap = encoded_output["heatmap"]
        size_map = encoded_output["size"].view(batch_size, 2, -1)        # NCHW to NC(HW)
        offset_map = encoded_output["offset"].view(batch_size, 2, -1)

        # obtain topk from heatmap
        heatmap = torch.sigmoid(encoded_output["heatmap"])  # convert to probability NOTE: is this necessary? sigmoid is a monotonic increasing function. max order will be preserved
        nms_mask = (heatmap == self.nms_max_pool(heatmap))  # pseudo-nms, only consider local peaks
        heatmap = nms_mask.float() * heatmap

        # flatten to N(CHW) to apply topk
        heatmap = heatmap.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(heatmap, num_detections)

        # restore flattened indices to class, xy indices
        topk_c_indices = topk_indices // (height*width)
        topk_xy_indices = topk_indices % (height*width)
        topk_y_indices = topk_xy_indices // width
        topk_x_indices = topk_xy_indices % width

        # extract bboxes at topk xy positions
        # bbox wh are already in relative scale
        # bbox xy offset are in absolute scale of output heatmap
        topk_w = torch.gather(size_map[:,0], dim=-1, index=topk_xy_indices)
        topk_h = torch.gather(size_map[:,1], dim=-1, index=topk_xy_indices)
        topk_x_offset = torch.gather(offset_map[:,0], dim=-1, index=topk_xy_indices)
        topk_y_offset = torch.gather(offset_map[:,1], dim=-1, index=topk_xy_indices)

        topk_x = (topk_x_indices + topk_x_offset) / width
        topk_y = (topk_y_indices + topk_y_offset) / height

        bboxes = torch.stack([topk_x, topk_y, topk_w, topk_h], dim=-1)  # NK4
        out = {
            "labels": topk_c_indices,
            "bboxes": bboxes,
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
            self.log(f"train_{k}_loss", v)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)

        total_loss = losses["heatmap"]
        for h in self.other_heads:
            total_loss += losses[h] * self.loss_weights[h]

        for k,v in losses.items():
            self.log(f"val_{k}_loss", v)
        self.log("val_total_loss", total_loss)

        pred_detections = self.decode_detections(encoded_output, num_detections=10)
        ap50, ar50 = self.evaluate_batch(pred_detections, batch)
        self.log("val_ap50", ap50)
        self.log("val_ar50", ar50)
        
        # only log sample images for the first validation batch
        if batch_idx == 0:
            imgs = batch["image"]
            num_samples = 8

            # draw bounding boxes on val images
            sample_imgs = self.draw_sample_images(imgs, pred_detections, batch, N_samples=num_samples)
            sample_imgs = sample_imgs.transpose(0,3,1,2)            # NHWC to NCHW
            sample_imgs = torchvision.utils.make_grid(torch.from_numpy(sample_imgs), nrow=num_samples)
            
            self.logger.experiment.add_image(
                "validation images", sample_imgs, 
                self.global_step, dataformats="chw")

            # log output heatmap
            pred_heatmap = torch.sigmoid(encoded_output["heatmap"][:num_samples])     # convert to probability
            pred_heatmap, _ = torch.max(pred_heatmap, dim=1)            # aggregate heatmaps across classes/channels
            pred_heatmap = pred_heatmap.cpu().numpy()
            pred_heatmap_scaled = pred_heatmap / np.max(pred_heatmap)   # scaled to [0,1]
            
            cm = plt.get_cmap("viridis")        # apply color map
            pred_heatmap = cm(pred_heatmap)[...,:3].transpose(0,3,1,2)      # NHWC to NCHW
            pred_heatmap_scaled = cm(pred_heatmap_scaled)[...,:3].transpose(0,3,1,2)

            pred_heatmap = torchvision.utils.make_grid(torch.from_numpy(pred_heatmap), nrow=num_samples)
            pred_heatmap_scaled = torchvision.utils.make_grid(torch.from_numpy(pred_heatmap_scaled), nrow=num_samples)

            self.logger.experiment.add_image(
                "predicted heatmap", pred_heatmap,
                self.global_step, dataformats="chw")
            
            self.logger.experiment.add_image(
                "predicted heatmap scaled", pred_heatmap_scaled,
                self.global_step, dataformats="chw")

            # log backbone feature map
            backbone_feature_map = encoded_output["backbone_features"][:num_samples]
            backbone_feature_map = torch.mean(backbone_feature_map, dim=1)
            backbone_feature_map = backbone_feature_map.cpu().numpy()
            backbone_feature_map = cm(backbone_feature_map)[...,:3].transpose(0,3,1,2)
            backbone_feature_map = torchvision.utils.make_grid(torch.from_numpy(backbone_feature_map), nrow=num_samples)
            self.logger.experiment.add_image(
                "backbone feature map", backbone_feature_map,
                self.global_step, dataformats="chw"
            )

    def test_step(self, batch, batch_idx):
        encoded_output = self(batch)
        losses = self.compute_loss(encoded_output, batch)

        pred_detections = self.decode_detections(encoded_output)
        ap50, ar50 = self.evaluate_batch(pred_detections, batch)
        self.log("test_ap50", ap50)
        self.log("test_ar50", ar50)

    @torch.no_grad()
    def evaluate_batch(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        # move to cpu and convert to numpy
        preds = {k: v.cpu().numpy() for k,v in preds.items()}
        targets = {k: v.cpu().numpy() for k,v in targets.items()}

        # convert cxcywh to x1y1x2y2
        convert_cxcywh_to_x1y1x2y2(preds["bboxes"])
        convert_cxcywh_to_x1y1x2y2(targets["bboxes"])

        ap50, ar50 = eval_detections(preds, targets, self.num_classes)
        return ap50, ar50

    @torch.no_grad()
    def draw_sample_images(
        self, imgs: torch.Tensor, preds: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor], N_samples: int=8
        ):
        indices = torch.arange(min(imgs.shape[0], N_samples))
        
        samples = imgs[indices].cpu().numpy().transpose(0,2,3,1)    # convert NCHW to NHWC
        samples = np.ascontiguousarray(samples[:,::2,::2,:])        # fast downsample via resampling  

        target_bboxes = targets["bboxes"].cpu().numpy()
        convert_cxcywh_to_x1y1x2y2(target_bboxes)
        target_labels = targets["labels"].cpu().numpy().astype(int)

        pred_bboxes = preds["bboxes"].cpu().numpy()
        convert_cxcywh_to_x1y1x2y2(pred_bboxes)
        pred_labels = preds["labels"].cpu().numpy().astype(int)
        pred_scores = preds["scores"].cpu().numpy()

        for i, idx in enumerate(indices):
            draw_bboxes(
                samples[i], pred_bboxes[idx], pred_labels[idx], pred_scores[idx], 
                inplace=True, relative_scale=True, color=RED)
    
            draw_bboxes(
                samples[i], target_bboxes[idx], target_labels[idx], 
                inplace=True, relative_scale=True, color=BLUE)

        return samples

    # lightning method
    def configure_optimizers(self):
        optimizer_algo = _optimizer_mapper.get(self.optimizer_name, torch.optim.Adam)
        optimizer = optimizer_algo(self.parameters(), lr=self.learning_rate)
        # lr scheduler
        return optimizer
