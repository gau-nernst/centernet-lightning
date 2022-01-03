import torch

from torchmetrics.detection.map import MAP

from .base import CenterNetBase
from .heads import HeatmapHead, Box2DHead
from ..utils import convert_cxcywh_to_xywh, convert_x1y1x2y2_to_xywh

class CenterNet(CenterNetBase):
    def __init__(
        self, 
        num_classes, 
        backbone, 
        neck, 
        head_width=256, 
        head_depth=1, 
        
        heatmap_prior=0.1,
        heatmap_method="cornernet",
        heatmap_loss="cornetnet_focal",
        
        box_loss="l1",

        **kwargs
        ):
        heads = {
            "heatmap": HeatmapHead(
                neck.out_channels, num_classes, width=head_width, depth=head_depth, 
                heatmap_prior=heatmap_prior, target_method=heatmap_method, loss_function=heatmap_loss
            ),
            "box_2d": Box2DHead(neck.out_channels, width=head_width, depth=head_depth, loss_function=box_loss)
        }
        super().__init__(backbone, neck, heads, **kwargs)
        self.metric = MAP()

    def validation_step(self, batch, batch_idx):
        encoded_outputs = self.get_encoded_outputs(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch)
        for k, v in losses.items():
            self.log(f"val/{k}_loss", v)
        
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#map
        mask = batch["mask"].cpu().numpy().astype(bool)
        bboxes = batch["bboxes"].cpu().numpy()

        bboxes = convert_cxcywh_to_xywh(bboxes)
        labels = batch["labels"].cpu().numpy()
        target = {
            "bboxes": [box[m] for box, m in zip(bboxes, mask)],    # list of 1-d np.ndarray of different lengths
            "labels": [label[m] for label, m in zip(labels, mask)]
        }
        
        preds = self.gather_detection2d(encoded_outputs["heatmap"].sigmoid(), encoded_outputs["box_2d"], normalize_bbox=True)
        preds = {k: v.cpu().numpy() for k,v in preds.items()}           # 2-d np array with dim batch_size x num_detections (100)
        preds["bboxes"] = convert_x1y1x2y2_to_xywh(preds["bboxes"])

        return preds, target
    

    def validation_epoch_end(self, outputs):
        preds, target = super().validation_epoch_end(outputs)
        
        metrics = evaluate_coco_detection(
            preds["bboxes"], preds["labels"], preds["scores"], 
            target["bboxes"], target["labels"],
            metrics_to_return=("AP", "AP50", "AP75")
        )
        
        for metric, value in metrics.items():
            self.log(f"val/{metric}", value)

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
