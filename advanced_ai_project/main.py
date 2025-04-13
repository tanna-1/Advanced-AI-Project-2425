from pathlib import Path
import argparse
import torch

from .hyperparameters import (
    load_hyperparameters,
    optimize_hyperparameters,
)
from .model import MLPCheckpoint
from .dataset import CharIndexDataset


def optimize_operation(
    dataset_path: str,
    checkpoint_path: str,
    study_path: str,
    batch_size: int,
    num_epochs: int,
    opt_trials: int,
    length_cutoff: int | None,
) -> None:
    if Path(checkpoint_path).exists():
        print(f"Checkpoint file exists. Hyperparameters will not be optimized.")
        return

    print("Optimizing hyperparameters...")
    try:
        optimize_hyperparameters(
            study_path,
            CharIndexDataset(dataset_path, length_cutoff=length_cutoff),
            n_trials=opt_trials,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
    except KeyboardInterrupt:
        print("Optimization interrupted.")


def train_operation(
    dataset_path: str,
    checkpoint_path: str,
    study_path: str,
    batch_size: int,
    num_epochs: int,
    length_cutoff: int | None,
) -> None:
    try:
        ckpt = MLPCheckpoint.load(checkpoint_path)
        print("Loaded existing checkpoint.")
    except:
        try:
            ckpt = MLPCheckpoint.new_from_hyperparams(load_hyperparameters(study_path))
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
            CharIndexDataset(dataset_path, length_cutoff=length_cutoff),
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        print(
            f"Training complete with an average loss of {avg_loss} over last 100 batches"
        )
    except KeyboardInterrupt:
        print("Training interrupted.")

    ckpt.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def evaluate_operation(checkpoint_path: str, start: int | None, count: int) -> None:
    print("Evaluating model...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    ckpt.model.eval()

    if start is None:
        start = ckpt.last_seen_index
    elif start < 0:
        start = ckpt.last_seen_index + start

    with torch.no_grad():

        def _eval(idx, length):
            inputs = torch.arange(idx, idx + length, device=ckpt.model.device)
            result = ckpt.model(inputs)

            predicted = [chr(int(x.item())) for x in torch.argmax(result, dim=1)]
            print(
                f"Predicted string from {idx} to {idx+length}: {repr(''.join(predicted))}"
            )

        _eval(start, count)


def main():
    parser = argparse.ArgumentParser(description="Advanced AI Project")
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoint.pt",
        help="Path to save or load the model checkpoint (default: checkpoint.pt)",
    )

    subparsers = parser.add_subparsers(
        dest="operation", required=True, help="Operation to perform"
    )

    # Subparser for 'optimize'
    optimize_parser = subparsers.add_parser("optimize", help="Optimize hyperparameters")
    optimize_parser.add_argument(
        "dataset_path",
        help="Path to the dataset (required for optimize)",
    )
    optimize_parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for optimization (default: 4096)",
    )
    optimize_parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of epochs for optimization (default: 2)",
    )
    optimize_parser.add_argument(
        "--opt-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimization (default: 100)",
    )
    optimize_parser.add_argument(
        "--length-cutoff",
        type=int,
        default=None,
        help="Maximum length of sequences in the dataset (default: None)",
    )
    optimize_parser.add_argument(
        "--study-path",
        default="study.db",
        help="Path to the database for storing optimization studies (default: study.db)",
    )

    # Subparser for 'train'
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "dataset_path",
        help="Path to the dataset (required for train)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training (default: 4096)",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of epochs for training (default: 2)",
    )
    train_parser.add_argument(
        "--length-cutoff",
        type=int,
        default=None,
        help="Maximum length of sequences in the dataset (default: None)",
    )
    train_parser.add_argument(
        "--study-path",
        default="study.db",
        help="Path to the database for loading hyperparameters (default: study.db)",
    )

    # Subparser for 'evaluate'
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting index for evaluation (default: None, uses last seen index)",
    )
    evaluate_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of characters to evaluate (default: 1000)",
    )

    args = parser.parse_args()

    if args.operation == "optimize":
        optimize_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
            args.opt_trials,
            args.length_cutoff,
        )
    elif args.operation == "train":
        train_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
            args.length_cutoff,
        )
    elif args.operation == "evaluate":
        evaluate_operation(args.checkpoint_path, args.start, args.count)
