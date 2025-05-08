from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from .utils import DEVICE
from .blocks import InvertedBottleneckMLP, BinaryEncode

# Model dimensions
INDEX_DIM = 64
OUT_DIM = 256

# Hidden width and depth are hardcoded as the optimizer always suggests the highest values
HIDDEN_WIDTH = 256
HIDDEN_DEPTH = 4


class MLPModel(nn.Module):
    def __init__(
        self,
        hidden_depth: int,
        hidden_width: int,
        expansion_factor: float,
        dropout: float,
        use_selu: bool,
        index_dim: int = INDEX_DIM,
        out_dim: int = OUT_DIM,
    ):
        super().__init__()

        self.seq = nn.Sequential(
            BinaryEncode(index_dim),
            nn.Linear(index_dim, hidden_width),
            *[
                InvertedBottleneckMLP(hidden_width, expansion_factor, dropout, use_selu)
                for _ in range(hidden_depth)
            ],
            nn.Linear(hidden_width, out_dim),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.seq(x)


@dataclass
class MLPCheckpoint:
    model: MLPModel
    optimizer: torch.optim.Optimizer
    hyperparameters: dict[str, Any]
    last_seen_index: int

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "hyperparameters": self.hyperparameters,
                "last_seen_index": self.last_seen_index,
            },
            path,
        )

    @staticmethod
    def load(path: str):
        values = torch.load(path)
        ckpt = MLPCheckpoint.new_from_hyperparams(values["hyperparameters"])

        ckpt.last_seen_index = values["last_seen_index"]
        ckpt.model.load_state_dict(values["model"])
        ckpt.optimizer.load_state_dict(values["optimizer"])
        return ckpt

    @staticmethod
    def new_from_hyperparams(hyperparameters: dict[str, Any]):
        model = MLPModel(
            hidden_depth=HIDDEN_DEPTH,
            hidden_width=HIDDEN_WIDTH,
            expansion_factor=hyperparameters["expansion_factor"],
            dropout=hyperparameters["dropout"],
            use_selu=hyperparameters["use_selu"],
        )
        model.to(DEVICE)

        return MLPCheckpoint(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=hyperparameters["lr"]),
            hyperparameters=hyperparameters,
            last_seen_index=0,
        )
