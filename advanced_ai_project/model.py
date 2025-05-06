import collections
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset.base import TokenDataset, TokenLatentDataset
from .utils import DEVICE
from .blocks import InvertedBottleneckMLP, BinaryEncode

# Model dimensions
INPUT_BIT_WIDTH = 64
HIDDEN_DEPTH = 16
HIDDEN_WIDTH = 512

# Output dimensions
IMAGE_DIM = 256
TOKEN_DIM = 257  # 256+1 for multimodal image output token! Last token is reserved for image output.
OUT_DIM = TOKEN_DIM + IMAGE_DIM

# Special tokens
SPECIAL_IMAGE_TOKEN = 256


class MultimodalLoss(nn.Module):
    def __init__(self, dataset_type: type[TokenDataset | TokenLatentDataset]):
        super().__init__()

        self.__dataset_type = dataset_type
        self.__ce_loss = nn.CrossEntropyLoss()
        self.__mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        batch_size = outputs.shape[0]
        loss = torch.zeros(1, device=outputs.device)

        # First TOKEN_DIM dimensions are token logits
        token_logits = outputs[:, :TOKEN_DIM]

        for i in range(batch_size):
            target = targets[i]

            if issubclass(self.__dataset_type, TokenDataset):
                # Apply cross-entropy over token_logits to predict text token
                loss = loss + self.__ce_loss(token_logits[i], target)

            elif issubclass(self.__dataset_type, TokenLatentDataset):
                # Apply cross-entropy over token_logits to predict token
                loss = loss + self.__ce_loss(token_logits[i], target[0].long())

                if int(target[0].item()) == SPECIAL_IMAGE_TOKEN:
                    # Apply L2 loss on the next dimensions for the image
                    image_output = outputs[i, TOKEN_DIM:]
                    image_target = target[1:]
                    loss = loss + self.__mse_loss(image_output, image_target)

        return loss / batch_size


class MLPModel(nn.Module):
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

        self.seq = nn.Sequential(
            BinaryEncode(input_bit_width),
            nn.Linear(input_bit_width, hidden_width),
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
        dataset: TokenDataset | TokenLatentDataset,
        num_epochs: int,
        batch_size: int,
        return_loss_over_n: int = 100,
        show_progress: bool = True,
    ):
        loss_fn = MultimodalLoss(type(dataset))

        # Last N losses
        loss_history = collections.deque(maxlen=return_loss_over_n)

        # shuffle=True causes issues with the lazy dataset
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
