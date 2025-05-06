from pathlib import Path
import argparse
import torch
from torchinfo import summary

from .hyperparameters import (
    load_hyperparameters,
    optimize_hyperparameters,
)
from .model import MLPCheckpoint
from .dataset import CharIndexDataset, StringDataset
from .evaluate import META_TRAINING_EPOCHS, evaluate


def optimize_operation(
    dataset_path: str,
    checkpoint_path: str,
    study_path: str,
    batch_size: int,
    num_epochs: int,
    opt_trials: int,
    length_cutoff: int | None,
):
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
):
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
            f"Training complete with an average loss of {avg_loss}"
        )
    except KeyboardInterrupt:
        print("Training interrupted.")

    ckpt.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def evaluate_operation(checkpoint_path: str, start: int | None, count: int):
    print("Evaluating model...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    if start is None:
        start = ckpt.last_seen_index
    elif start < 0:
        start = ckpt.last_seen_index + start
    print(evaluate(ckpt, start, count))


def autocomplete_operation(checkpoint_path: str, prompt: str, count: int):
    print("Autocompleting via model...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    start_index = ckpt.last_seen_index + 1

    print("Training on the prompt...")
    ckpt.model.train()
    avg_loss = ckpt.train(
        StringDataset(prompt, index_offset=start_index),
        num_epochs=META_TRAINING_EPOCHS,
        batch_size=1,
    )
    print(f"Prompt training complete with an average loss of {avg_loss}")

    print(evaluate(ckpt, start_index + len(prompt), count))


def model_info_operation(checkpoint_path: str, batch_size: int):
    print("Displaying model summary...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    summary(
        ckpt.model,
        input_data=torch.zeros(
            (batch_size, 64), dtype=torch.int64, device=ckpt.model.device
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Advanced AI Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoint.pt",
        help="Path to save or load the model checkpoint",
    )
    subparsers = parser.add_subparsers(
        dest="operation", required=True, help="Operation to perform"
    )

    optimize_parser = subparsers.add_parser("optimize", help="Optimize hyperparameters")
    optimize_parser.add_argument("dataset_path", help="Path to the dataset")
    optimize_parser.add_argument(
        "--opt-trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimization",
    )
    optimize_parser.add_argument(
        "--study-path",
        default="study.db",
        help="Path to the database for storing optimization studies",
    )
    optimize_parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size"
    )
    optimize_parser.add_argument(
        "--num-epochs", type=int, default=2, help="Number of epochs"
    )
    optimize_parser.add_argument(
        "--length-cutoff",
        type=int,
        default=None,
        help="Maximum length of sequences in the dataset",
    )

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("dataset_path", help="Path to the dataset")
    train_parser.add_argument(
        "--study-path",
        default="study.db",
        help="Path to the database for loading hyperparameters",
    )
    train_parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    train_parser.add_argument(
        "--num-epochs", type=int, default=2, help="Number of epochs"
    )
    train_parser.add_argument(
        "--length-cutoff",
        type=int,
        default=None,
        help="Maximum length of sequences in the dataset",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument(
        "--start", type=int, default=None, help="Starting index for evaluation"
    )
    evaluate_parser.add_argument(
        "--count", type=int, default=100, help="Number of characters to evaluate"
    )

    autocomplete_parser = subparsers.add_parser(
        "autocomplete", help="Autocomplete using the model"
    )
    autocomplete_parser.add_argument("prompt", help="Prompt string to autocomplete")
    autocomplete_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of characters to predict after the prompt",
    )

    model_info_parser = subparsers.add_parser(
        "model_info", help="Display model summary"
    )
    model_info_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for the model summary"
    )

    args = parser.parse_args()
    operations = {
        "optimize": lambda: optimize_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
            args.opt_trials,
            args.length_cutoff,
        ),
        "train": lambda: train_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
            args.length_cutoff,
        ),
        "evaluate": lambda: evaluate_operation(
            args.checkpoint_path, args.start, args.count
        ),
        "autocomplete": lambda: autocomplete_operation(
            args.checkpoint_path,
            args.prompt,
            args.count,
        ),
        "model_info": lambda: model_info_operation(
            args.checkpoint_path, args.batch_size
        ),
    }
    operations[args.operation]()
