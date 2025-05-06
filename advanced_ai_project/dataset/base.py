import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


# TokenDataset yields a token and its corresponding index
class TokenDataset(Dataset[tuple[int, int]], ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[int, int]: ...


# TokenLatentDataset yields a token with its latent data concatenated, and its corresponding index
class TokenLatentDataset(Dataset[tuple[int, torch.Tensor]], ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]: ...
