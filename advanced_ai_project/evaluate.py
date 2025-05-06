from dataclasses import dataclass
from typing import Sequence
import torch

from .model import MLPModel, TOKEN_DIM, SPECIAL_IMAGE_TOKEN


@dataclass
class TextToken:
    char: str


@dataclass
class ImageToken:
    image: torch.Tensor


def evaluate(
    model: MLPModel, idx: int, length: int
) -> Sequence[TextToken | ImageToken]:
    with torch.no_grad():
        model.eval()
        inputs = torch.arange(idx, idx + length, device=model.device)
        result = model(inputs)

        predicted = []
        for i in range(result.size(0)):
            token_logits = result[i, :TOKEN_DIM]
            token_id = torch.argmax(token_logits).item()

            if token_id == SPECIAL_IMAGE_TOKEN:
                # Extract image data from the remaining dimensions
                image_data = result[i, TOKEN_DIM:]
                predicted.append(ImageToken(image_data))
            else:
                predicted.append(TextToken(chr(int(token_id))))

        return predicted
