from typing import Tuple, Dict, Union

import torch
import albumentations as A

class TrivialAugmentWide(A.Sequential):
    def __init__(self):
        transforms = [

        ]
        super().__init__(transforms)


class Mosaic:
    def __init__(self):
        pass

    def __call__(self, images: torch.Tensor, targets: Tuple[Dict[str, Union[Tuple, int]]]):
        pass
