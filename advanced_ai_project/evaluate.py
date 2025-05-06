from dataclasses import dataclass
from typing import Sequence
import torch
from tqdm import tqdm

from .dataset import SingleTokenDataset
from .model import SPECIAL_IMAGE_TOKEN, TOKEN_DIM, MLPCheckpoint

META_TRAINING_EPOCHS = 1


@dataclass
class TextToken:
    char: str


@dataclass
class ImageToken:
    image: torch.Tensor


def evaluate(
    ckpt: MLPCheckpoint, idx: int, length: int
) -> Sequence[TextToken | ImageToken]:
    predicted = []

    ckpt.model.train()
    for i in tqdm(range(idx, idx + length)):
        result = ckpt.model(torch.tensor([i], device=ckpt.model.device))
        token_logits = result[0][:TOKEN_DIM]
        sampled_token = torch.argmax(token_logits)

        ckpt.train(
            SingleTokenDataset(sampled_token, i),
            num_epochs=META_TRAINING_EPOCHS,
            batch_size=1,
            show_progress=False,
        )

        if sampled_token == SPECIAL_IMAGE_TOKEN:
            # Extract image data from the remaining dimensions
            image_data = result[i, TOKEN_DIM:]
            predicted.append(ImageToken(image_data))
        else:
            predicted.append(TextToken(chr(int(sampled_token))))

    return predicted
