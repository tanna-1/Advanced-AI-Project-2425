from dataclasses import dataclass
from typing import Any, Generator, Iterable
import torch

from .utils import sample_top_k
from .dataset.text import SingleTokenDataset
from .model import SPECIAL_IMAGE_TOKEN, TOKEN_DIM, MLPCheckpoint

META_TRAINING_EPOCHS = 1


@dataclass
class TextToken:
    char: str


@dataclass
class ImageToken:
    image: torch.Tensor


def print_tokens(
    tokens: Iterable[TextToken | ImageToken],
):
    for token in tokens:
        if isinstance(token, ImageToken):
            print(f"<image/>", end="", flush=True)
        elif isinstance(token, TextToken):
            c = token.char
            c = c if c.isprintable() or c in "\r\n" else f"\\x{ord(c):02x}"
            print(c, end="", flush=True)
        else:
            print(f"<unknown type={type(token)}/>", end="", flush=True)


def evaluate(
    ckpt: MLPCheckpoint,
    idx: int,
    length: int,
    temperature: float = 1.0,
    top_k: int = 5,
) -> Generator[TextToken | ImageToken, Any, None]:
    ckpt.model.train()
    for i in range(idx, idx + length):
        # Input is not batched, so get the first dimension of the output
        output = ckpt.model(torch.tensor([i], device=ckpt.model.device))[0]

        # Sample a token from the token logits
        sample = sample_top_k(
            logits=output[:TOKEN_DIM], k=top_k, temperature=temperature
        )
        sample = int(sample.item())

        ckpt.train(
            SingleTokenDataset(sample, i),
            num_epochs=META_TRAINING_EPOCHS,
            batch_size=1,
            show_progress=False,
        )

        if sample == SPECIAL_IMAGE_TOKEN:
            # Extract image data from the remaining dimensions
            yield ImageToken(output[TOKEN_DIM:])
        else:
            yield TextToken(chr(sample))
