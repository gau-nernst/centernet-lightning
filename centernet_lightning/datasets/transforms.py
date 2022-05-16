from typing import Tuple, Dict, Union
from copy import deepcopy

import torch
import albumentations as A


# https://arxiv.org/abs/2103.10158
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/autoaugment.py
class TrivialAugmentWide(A.OneOf):
    def __init__(self):
        transforms = [
            A.Affine(shear={'x': (-45,45), 'y': 0}),
            A.Affine(shear={'x': 0, 'y': (-45,45)}),
            A.Affine(translate_px={'x': (-32,32), 'y': 0}),
            A.Affine(translate_px={'x': 0, 'y': (-32,32)}),
            A.Affine(rotate=(-135,135)),
            A.ColorJitter(brightness=0.99, contrast=0, saturation=0, hue=0),
            A.ColorJitter(brightness=0, contrast=0.99, saturation=0, hue=0),
            A.ColorJitter(brightness=0, contrast=0, saturation=0.99, hue=0),
            A.Sharpen(alpha=(0, 0.99), lightness=(1,1)),
            A.Posterize(num_bits=(2,8)),
            A.Solarize(threshold=(0,255)),
            # no autoconstrast
            A.Equalize()
        ]
        num_t = len(transforms)
        super().__init__(transforms, p=num_t/(num_t+1))


def mosaic_2x2(images: torch.Tensor, targets: Tuple[Dict[str, Union[Tuple, int]]]):
    N, C, H, W = images.shape
    N_4 = N // 4

    # 0 | 1
    # --+--
    # 2 | 3
    mosaic = torch.zeros((N_4, C, H*2, W*2), dtype=images.dtype, device=images.device)
    mosaic[:,:,:H,:W] = images[:N_4]
    mosaic[:,:,:H,W:] = images[N_4:N_4*2]
    mosaic[:,:,H:,:W] = images[N_4*2:N_4*3]
    mosaic[:,:,H:,W:] = images[N_4*3:]

    mosaic_targets = []
    box_offsets = ((0,0), (W,0), (0, H), (W,H))
    targets = deepcopy(targets)
    for i in range(N_4):
        labels = []
        boxes = []
        for j, (x_offset, y_offset) in enumerate(box_offsets):
            labels.extend(targets[i+N_4*j]['labels'])
            for x, y, w, h in targets[i+N_4*j]['boxes']:
                boxes.append((x+x_offset, y+y_offset, w, h))

        mosaic_targets.append({'boxes': boxes, 'labels': labels})

    return mosaic, tuple(mosaic_targets)
