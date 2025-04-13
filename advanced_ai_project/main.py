from pathlib import Path
import sys
import torch

from .hyperparameters import (
    load_hyperparameters,
    optimize_hyperparameters,
)
from .model import MLPCheckpoint
from .dataset import CharIndexDataset
from .utils import DEVICE


# Hardcoded parameters
CHECKPOINT_PATH = "checkpoint.pt"
BATCH_SIZE = 4096
NUM_EPOCHS = 2
OPT_TRIALS = 100


def evaluate_model(ckpt: MLPCheckpoint):
    ckpt.model.eval()

    with torch.no_grad():

        def _eval(idx, length):
            inputs = torch.arange(idx, idx + length, device=DEVICE)
            result = ckpt.model(inputs)

            predicted = [chr(int(x.item())) for x in torch.argmax(result, dim=1)]
            print(
                f"Predicted string from {idx} to {idx+length}: {repr(''.join(predicted))}"
            )

        last_idx = ckpt.last_seen_index
        _eval(0, 100)
        _eval(last_idx - 100, 100)


def main():
    operation = sys.argv[1] if len(sys.argv) > 1 else None
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else None

    if operation == "optimize":
        if dataset_path is None:
            print("Please provide a dataset path.")
            return

        if Path(CHECKPOINT_PATH).exists():
            print(f"Checkpoint file exists. Hyperparameters will not be optimized.")
            return

        print("Optimizing hyperparameters...")
        try:
            optimize_hyperparameters(
                CharIndexDataset(dataset_path),
                n_trials=OPT_TRIALS,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
            )
        except KeyboardInterrupt:
            print("Optimization interrupted.")
    elif operation == "train":
        if dataset_path is None:
            print("Please provide a dataset path.")
            return

        try:
            ckpt = MLPCheckpoint.load(CHECKPOINT_PATH)
            print("Loaded existing checkpoint.")
        except:
            try:
                ckpt = MLPCheckpoint.new_from_hyperparams(load_hyperparameters())
                print("Created new checkpoint from hyperparameters.")
            except:
                print(
                    f"Neither checkpoint or the hyperparameter DB exists. Please run 'optimize' first."
                )
                return

        print("Training model...")
        ckpt.model.train()

        try:
            avg_loss = ckpt.train(
                CharIndexDataset(dataset_path),
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
            )
            print(
                f"Training complete with an average loss of {avg_loss} over last 100 batches"
            )
        except KeyboardInterrupt:
            print("Training interrupted.")

        ckpt.save(CHECKPOINT_PATH)
        print(f"Model saved to {CHECKPOINT_PATH}")
    elif operation == "evaluate":
        print("Evaluating model...")
        evaluate_model(MLPCheckpoint.load(CHECKPOINT_PATH))
    else:
        print(f"Usage: python {sys.argv[0]} [optimize|train|evaluate]")
