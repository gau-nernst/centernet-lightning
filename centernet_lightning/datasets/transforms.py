from typing import Tuple, Dict, Union

import torch
import albumentations as A


# https://github.com/pytorch/vision/blob/main/torchvision/transforms/autoaugment.py
class TrivialAugmentWide(A.OneOf):
    def __init__(self):
        transforms = [
            A.Affine(shear={'x': 45, 'y': 0}),
            A.Affine(shear={'x': 0, 'y': 45}),
            A.Affine(translate_px={'x': 32, 'y': 0}),
            A.Affine(translate_px={'x': 0, 'y': 32}),
            A.Affine(rotate=135),
            A.ColorJitter(brightness=0.99, contrast=0, saturation=0, hue=0),
            A.ColorJitter(brightness=0, contrast=0.99, saturation=0, hue=0),
            A.ColorJitter(brightness=0, contrast=0, saturation=0.99, hue=0),
            A.Sharpen(alpha=(0, 0.99), lightness=(1,1)),
            A.Posterize(num_bits=(2, 8)),
            A.Solarize(threshold=(0, 255)),
            # no autoconstrast
            A.Equalize()
        ]
        num_t = len(transforms)
        super().__init__(transforms, p=num_t/(num_t+1))


class Mosaic:
    def __init__(self):
        pass

    def __call__(self, images: torch.Tensor, targets: Tuple[Dict[str, Union[Tuple, int]]]):
        pass
