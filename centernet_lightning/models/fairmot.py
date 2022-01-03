import torch

from .base import CenterNetBase
from .tracker import Tracker
from ..utils import convert_cxcywh_to_xywh, convert_x1y1x2y2_to_xywh
from ..eval import evaluate_mot_tracking_sequence


class FairMOT(CenterNetBase):
    # rank 1 only?
    def on_validation_epoch_start(self):
        self.tracker = Tracker(device=self.device)
    
    def validation_step(self, batch, batch_idx):
        encoded_outputs = self.get_encoded_outputs(batch["image"])
        losses = self.compute_loss(encoded_outputs, batch, ignore_reid=True)    # during validation, only evaluate detection loss
        for k,v in losses.items():
            self.log(f"val/{k}_loss", v)
        
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#map
        mask = batch["mask"].cpu().numpy().astype(bool)
        bboxes = batch["bboxes"].cpu().numpy()

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
        preds, target = super().validation_epoch_end(outputs)

        metrics = evaluate_mot_tracking_sequence(
            preds["bboxes"], preds["track_ids"],
            target["bboxes"], target["track_ids"]
        )
        self.tracker = None
    
        for metric, value in metrics.items():
            self.log(f"val/{metric}", value)

    def gather_tracking2d(self, heatmap: torch.Tensor, box_2d: torch.Tensor, reid: torch.Tensor, num_detections=100, nms_kernel=3, normalize_bbox=False):
        """Decode model outputs for tracking task
        """
        topk_scores, topk_indices, topk_labels = HeatmapHead.gather_topk(heatmap, nms_kernel=nms_kernel, num_detections=num_detections)
        topk_bboxes = Box2DHead.gather_at_indices(box_2d, topk_indices, normalize_bbox=normalize_bbox, stride=self.output_stride)
        topk_embeddings = EmbeddingHead.gather_at_indices(reid, topk_indices)

        out = {
            "bboxes": topk_bboxes,
            "labels": topk_labels,
            "scores": topk_scores,
            "embeddings": topk_embeddings
        }
        return out

    
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

