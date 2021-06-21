from typing import Dict, Iterable

import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def collate_detections_with_padding(batch: Iterable[Dict[str, np.ndarray]], pad_value: int = 0):
    output = {key: [] for key in batch[0]}
    output["mask"] = []
    max_size = 0

    # collate items to a list and find max number of detection in this batch
    for item in batch:
        for key, value in item.items():
            output[key].append(value)
        max_size = max(max_size, len(item["labels"]))
    
    # pad labels and masks to max length
    for i in range(len(batch)):
        item_size = len(output["labels"][i])
        output["mask"].append([1]*item_size)

        for _ in range(max_size - item_size):
            output["bboxes"][i].append([pad_value]*4)
            output["labels"][i].append(pad_value)
            output["mask"][i].append(0)    
    
    # image is a list of tensor -> use torch.stack
    # the rest are nested lists -> use torch.tensor
    for key, value in output.items():
        if key != "image":
            output[key] = torch.tensor(value)
        else:
            output[key] = torch.stack(value, dim=0)

    return output