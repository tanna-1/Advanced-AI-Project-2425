import re
import torch
import torch.nn as nn
from tqdm import tqdm

from .encoders import binary_encode_numbers
from .blocks import InvertedBottleneckMLP
from .utils import get_torch_device


class MLPLangModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_depth: int,
        hidden_width: int,
        out_dim: int,
        expansion_factor: float = 4,
        dropout: float = 0.0,
        use_selu: bool = False,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
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
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="selu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.seq(x)


out_dim = 256
input_dim = 64
batch_size = 32
num_epochs = 2000
lr = 1e-3

device = get_torch_device()
model = MLPLangModel(input_dim, 8, 256, out_dim, 4, 0.1, True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


def main():
    loss = None
    for _ in tqdm(range(num_epochs)):
        inputs = torch.randint(0, out_dim, (batch_size,)).to(device)
        targets = inputs.clone()
        binary_inputs = binary_encode_numbers(inputs, input_dim)

        result = model(binary_inputs)
        loss = loss_fn(result, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if loss:
        print(f"Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        try:
            while True:
                num_input = torch.tensor(int(input("Enter a number: ")), device=device)
                encoded_input = binary_encode_numbers(num_input, input_dim)
                result = model(encoded_input)

                predicted = torch.argmax(result, dim=0)
                print(f"Actual: {num_input.item()}")
                print(f"Encoded Input: {encoded_input}")
                print(f"Predicted: {predicted.item()}")
        except (KeyboardInterrupt, EOFError):
            pass
