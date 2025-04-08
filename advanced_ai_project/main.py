import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CharIndexDataset
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
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.seq(x)


def main():
    out_dim = 256
    input_dim = 64
    batch_size = 4096
    num_epochs = 2
    lr = 1e-3

    device = get_torch_device()
    model = MLPLangModel(input_dim, 8, 256, out_dim, 4, 0.1, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = CharIndexDataset("../item.csv", 20_000_0)

    # Last 10 losses
    loss_history = collections.deque(maxlen=100)

    # Shuffle is enabled for now, need to check if it's useful
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    for _ in tqdm(range(num_epochs)):
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            encoded_inputs = binary_encode_numbers(inputs, input_dim)
            result = model(encoded_inputs)

            loss = loss_fn(result, targets)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(
        f"Average loss over the last {len(loss_history)} batches: {sum(loss_history) / len(loss_history)}"
    )

    model.eval()
    with torch.no_grad():

        def evaluate(idx, length):
            inputs = torch.arange(idx, idx + length, device=device)
            encoded_input = binary_encode_numbers(inputs, input_dim)
            result = model(encoded_input)

            predicted = [chr(int(x.item())) for x in torch.argmax(result, dim=1)]
            print(
                f"Predicted string from {idx} to {idx+length}: {repr(''.join(predicted))}"
            )

        last_idx = len(dataset) - 1
        evaluate(0, 100)
        evaluate(last_idx - 100, 100)
