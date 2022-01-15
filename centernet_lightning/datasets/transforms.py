from typing import Tuple, Dict, Union

import torch


class Mosaic:
    def __init__(self):
        super().__init__()

    def __call__(self, images: torch.Tensor, targets: Tuple[Dict[str, Union[Tuple, int]]]):
        pass
