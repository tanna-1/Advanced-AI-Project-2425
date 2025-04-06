import torch.nn as nn
from .dropout import ConditionalAlphaDropout, ConditionalDropout


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


# Residual MLP Block
class PlainMLP(nn.Module):
    def __init__(
        self,
        dim: int,
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

        self.fn = nn.Sequential(
            nn.Linear(dim, dim),
            activation_fn(),
            dropout_fn(dropout),
        )
        self.ln = layer_norm

    def forward(self, x):
        return x + self.fn(self.ln(x))
