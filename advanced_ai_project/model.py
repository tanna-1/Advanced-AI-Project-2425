import collections
from dataclasses import dataclass
from typing import Any
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import DEVICE
from .blocks import InvertedBottleneckMLP, BinaryEncode

# Hardcoded model parameters
OUT_DIM = 256
INPUT_BIT_WIDTH = 64
HIDDEN_DEPTH = 16
HIDDEN_WIDTH = 512

# Display parameters
INNER_PROGRESS_CUTOFF = 1000


class MLPLangModel(nn.Module):
    def __init__(
        self,
        input_bit_width: int = INPUT_BIT_WIDTH,
        hidden_depth: int = HIDDEN_DEPTH,
        hidden_width: int = HIDDEN_WIDTH,
        out_dim: int = OUT_DIM,
        expansion_factor: float = 4,
        dropout: float = 0.0,
        use_selu: bool = False,
    ):
        super().__init__()

        # Hidden depth is reduced based on the expansion factor to avoid incorrect optimization of hyperparameters
        hidden_depth = int(hidden_depth // (expansion_factor + 1))

        self.seq = nn.Sequential(
            BinaryEncode(input_bit_width),
            nn.Linear(input_bit_width, hidden_width),
            *[
                InvertedBottleneckMLP(hidden_width, expansion_factor, dropout, use_selu)
                for _ in range(hidden_depth)
            ],
            nn.Linear(hidden_width, out_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.seq(x)


@dataclass
class MLPCheckpoint:
    model: MLPLangModel
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
        model = MLPLangModel(
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
    ):
        loss_fn = nn.CrossEntropyLoss()

        # Last N losses
        loss_history = collections.deque(maxlen=return_loss_over_n)

        # Shuffle is disabled for easier training continuation
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, shuffle=False
        )

        for _ in tqdm(range(num_epochs)):
            if len(dataloader) > INNER_PROGRESS_CUTOFF:
                iterator = tqdm(dataloader)
            else:
                iterator = dataloader

            for inputs, targets in iterator:
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
