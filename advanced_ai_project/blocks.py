import torch
import torch.nn as nn
from .dropout import ConditionalAlphaDropout, ConditionalDropout


class BinaryEncode(nn.Module):
    def __init__(self, num_bits: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.__num_bits = num_bits
        self.__dtype = dtype

    def forward(self, x):
        arange = torch.arange(self.__num_bits, device=x.device)
        return ((x.unsqueeze(-1) & (1 << arange)) > 0).to(dtype=self.__dtype)


# Residual Inverted Bottleneck Block
class InvertedBottleneckMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 4,
        dropout: float = 0.0,
        use_selu: bool = False,
    ):
        super().__init__()

        if use_selu:
            # Same as the original, but with SELU activation,
            # Alpha Dropout instead of Dropout and no Layer Norm.
            (activation_fn, dropout_fn, layer_norm) = (
                nn.SELU,
                ConditionalAlphaDropout,
                nn.Identity(),
            )
        else:
            (activation_fn, dropout_fn, layer_norm) = (
                nn.GELU,
                ConditionalDropout,
                nn.LayerNorm(dim),
            )

        expanded_dim = int(expansion_factor * dim)
        self.fn = nn.Sequential(
            nn.Linear(dim, expanded_dim),
            activation_fn(),
            dropout_fn(dropout),
            nn.Linear(expanded_dim, dim),
            dropout_fn(dropout),
        )
        self.ln = layer_norm

    def forward(self, x):
        return x + self.fn(self.ln(x))
