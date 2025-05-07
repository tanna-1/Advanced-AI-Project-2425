import collections
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.data import Dataset
from .utils import DEVICE
from .blocks import InvertedBottleneckMLP, BinaryEncode

# Model dimensions
INPUT_DIM = 64
OUT_DIM = 256


class MLPModel(nn.Module):
    def __init__(
        self,
        hidden_depth: int,
        hidden_width: int,
        expansion_factor: float,
        dropout: float,
        use_selu: bool,
        input_dim: int = INPUT_DIM,
        out_dim: int = OUT_DIM,
    ):
        super().__init__()

        self.seq = nn.Sequential(
            BinaryEncode(input_dim),
            nn.Linear(input_dim, hidden_width),
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
            hidden_depth=hyperparameters["hidden_depth"],
            hidden_width=hyperparameters["hidden_width"],
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

    # Returns the average loss over the last N batches
    def train(
        self,
        dataset: Dataset,
        num_epochs: int,
        batch_size: int,
        return_loss_over_n: int = 100,
        show_progress: bool = True,
    ):
        loss_fn = nn.CrossEntropyLoss()

        # Last N losses
        loss_history = collections.deque(maxlen=return_loss_over_n)

        # shuffle=True causes issues with lazy datasets
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        progress = lambda x: tqdm(x) if show_progress else x
        for _ in progress(range(num_epochs)):
            for inputs, targets in progress(dataloader):
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device)

                self.last_seen_index = inputs.max().item()

                result = self.model(inputs)

                loss = loss_fn(result, targets)
                loss_history.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return sum(loss_history) / len(loss_history)
