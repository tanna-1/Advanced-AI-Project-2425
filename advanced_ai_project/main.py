import collections
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CharIndexDataset
from .blocks import InvertedBottleneckMLP, BinaryEncode
from .utils import get_torch_device

OUT_DIM = 256
INPUT_BIT_WIDTH = 64
DEVICE = get_torch_device()
DATASET = CharIndexDataset("../item.csv", 20_000_000)


class MLPLangModel(nn.Module):
    def __init__(
        self,
        input_bit_width: int,
        hidden_depth: int,
        hidden_width: int,
        out_dim: int,
        expansion_factor: float = 4,
        dropout: float = 0.0,
        use_selu: bool = False,
    ):
        super().__init__()

        # Hidden depth is reduced based on the expansion factor to avoid incorrect optimization of hyperparameters
        hidden_depth = int(hidden_depth // (expansion_factor + 1))

        self.seq = nn.Sequential(
            BinaryEncode(input_bit_width),
            nn.Linear(input_bit_width, hidden_width),
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


# Returns the average loss over the last N batches
def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    return_loss_over_n: int = 100,
):
    loss_fn = nn.CrossEntropyLoss()

    # Last N losses
    loss_history = collections.deque(maxlen=return_loss_over_n)

    # Shuffle is enabled for now, need to check if it's useful
    dataloader = DataLoader(DATASET, batch_size=batch_size, pin_memory=True)

    for _ in tqdm(range(num_epochs)):
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            result = model(inputs)

            loss = loss_fn(result, targets)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return sum(loss_history) / len(loss_history)


def evaluate_model(model: nn.Module):
    model.eval()
    with torch.no_grad():

        def _eval(idx, length):
            inputs = torch.arange(idx, idx + length, device=DEVICE)
            result = model(inputs)

            predicted = [chr(int(x.item())) for x in torch.argmax(result, dim=1)]
            print(
                f"Predicted string from {idx} to {idx+length}: {repr(''.join(predicted))}"
            )

        last_idx = len(DATASET) - 1
        _eval(0, 100)
        _eval(last_idx - 100, 100)


def optuna_objective(trial: optuna.Trial) -> float:
    # Fix the random seed for optimization of hyperparameters
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    model = MLPLangModel(
        input_bit_width=INPUT_BIT_WIDTH,
        hidden_depth=8,
        hidden_width=256,
        out_dim=OUT_DIM,
        expansion_factor=trial.suggest_float("expansion_factor", 1.0, 4.0, step=0.5),
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        use_selu=trial.suggest_categorical("use_selu", [True, False]),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-3)
    )

    return train_model(
        model,
        optimizer,
        num_epochs=2,
        batch_size=trial.suggest_int("batch_size", 32, 8192),
        return_loss_over_n=100,
    )


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=100)
    print(
        f"Hyperparameter optimization complete. Training with params: {study.best_params}"
    )

    model = MLPLangModel(
        input_bit_width=INPUT_BIT_WIDTH,
        hidden_depth=8,
        hidden_width=256,
        out_dim=OUT_DIM,
        expansion_factor=study.best_params["expansion_factor"],
        dropout=study.best_params["dropout"],
        use_selu=study.best_params["use_selu"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.best_params["lr"])

    avg_loss = train_model(
        model,
        optimizer,
        num_epochs=10,
        batch_size=study.best_params["batch_size"],
    )
    print(f"Training complete with an average loss of {avg_loss} over last 100 batches")

    torch.save(model, "tensor.pt")
    print("Model saved to tensor.pt. Evaluating model...")

    evaluate_model(model)
