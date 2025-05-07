from typing import Any, Generator, Iterable
import torch

from ..model import MLPCheckpoint
from ..utils import sample_top_k
from .train import train
from .dataset import SingleTokenDataset


META_TRAINING_EPOCHS = 1


def print_tokens(
    tokens: Iterable[str],
):
    for token in tokens:
        token = (
            token if token.isprintable() or token in "\r\n" else f"\\x{ord(token):02x}"
        )
        print(token, end="", flush=True)


def evaluate(
    ckpt: MLPCheckpoint,
    idx: int,
    length: int,
    temperature: float = 1.0,
    top_k: int = 5,
) -> Generator[str, Any, None]:
    ckpt.model.train()
    for i in range(idx, idx + length):
        # Input is not batched, so get the first dimension of the output
        output = ckpt.model(torch.tensor([i], device=ckpt.model.device))[0]

        # Sample a token from the logits
        sample = sample_top_k(logits=output, k=top_k, temperature=temperature)
        sample = int(sample.item())

        train(
            ckpt,
            SingleTokenDataset(sample, i),
            num_epochs=META_TRAINING_EPOCHS,
            batch_size=1,
            show_progress=False,
        )

        yield chr(sample)
