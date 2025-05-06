import torch

from .base import TokenLatentDataset
from ..model import IMAGE_DIM, SPECIAL_IMAGE_TOKEN


# Yields an image token and subsequently the ASCII digit
class ExampleImageDataset(TokenLatentDataset):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        if idx % 2 == 0:
            return idx, torch.cat(
                [torch.tensor([SPECIAL_IMAGE_TOKEN]), torch.zeros(IMAGE_DIM)]
            )
        else:
            return idx, torch.cat([torch.tensor([ord("0")]), torch.zeros(IMAGE_DIM)])
