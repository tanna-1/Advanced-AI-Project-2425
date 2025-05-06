from dataclasses import dataclass
from typing import Any, Generator, Iterable
import torch

from .dataset import SingleTokenDataset
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
            print(token.char.encode("unicode_escape").decode(), end="", flush=True)
        else:
            print(f"<unknown type={type(token)}/>", end="", flush=True)


def evaluate(
    ckpt: MLPCheckpoint, idx: int, length: int
) -> Generator[TextToken | ImageToken, Any, None]:
    ckpt.model.train()
    for i in range(idx, idx + length):
        # Input is not batched, so get the first dimension of the output
        output = ckpt.model(torch.tensor([i], device=ckpt.model.device))[0]

        token_logits = output[:TOKEN_DIM]
        sampled_token = torch.argmax(token_logits)

        ckpt.train(
            SingleTokenDataset(sampled_token, i),
            num_epochs=META_TRAINING_EPOCHS,
            batch_size=1,
            show_progress=False,
        )

        if sampled_token == SPECIAL_IMAGE_TOKEN:
            # Extract image data from the remaining dimensions
            image_data = output[TOKEN_DIM:]
            yield ImageToken(image_data)
        else:
            ascii = chr(int(sampled_token))
            yield TextToken(ascii)
