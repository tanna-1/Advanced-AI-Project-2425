from pathlib import Path
import argparse
import torch
from torchinfo import summary

from advanced_ai_project.utils import get_cifar10_dataset

from .text_prediction.train import train
from .hyperparameters import (
    load_hyperparameters,
    optimize_hyperparameters,
)
from .model import MLPCheckpoint
from .text_prediction.dataset import ByteFileDataset, StringDataset
from .text_prediction.evaluate import META_TRAINING_EPOCHS, evaluate, print_tokens
from .hypernet.train import train as train_hypernet
from .hypernet.evaluate import evaluate as evaluate_hypernet


def optimize_operation(
    dataset_path: str,
    checkpoint_path: str,
    study_path: str,
    batch_size: int,
    num_epochs: int,
    trials: int,
    length_cutoff: int | None,
):
    if Path(checkpoint_path).exists():
        print(f"Checkpoint file exists. Hyperparameters will not be optimized.")
        return

    print("Optimizing hyperparameters...")
    try:
        optimize_hyperparameters(
            study_path,
            ByteFileDataset(dataset_path, length_cutoff=length_cutoff),
            n_trials=trials,
            num_epochs=num_epochs,
            batch_size=batch_size,
            train_model="text_prediction",
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
        avg_loss = train(
            ckpt,
            ByteFileDataset(dataset_path, length_cutoff=length_cutoff),
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
        print(f"Training complete with an average loss of {avg_loss}")
    except KeyboardInterrupt:
        print("Training interrupted.")

    ckpt.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


def evaluate_operation(
    checkpoint_path: str, start: int | None, length: int, temperature: float, top_k: int
):
    print("Evaluating model...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    if start is None:
        start = ckpt.last_seen_index
    elif start < 0:
        start = ckpt.last_seen_index + start

    print_tokens(evaluate(ckpt, start, length, temperature=temperature, top_k=top_k))


def autocomplete_operation(
    checkpoint_path: str, prompt: str, length: int, temperature: float, top_k: int
):
    print("Autocompleting via model...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    start_index = ckpt.last_seen_index + 1

    print("Training on the prompt...")
    ckpt.model.train()
    avg_loss = train(
        ckpt,
        StringDataset(prompt, start_index=start_index),
        num_epochs=META_TRAINING_EPOCHS,
        batch_size=1,
    )
    print(f"Prompt training complete with an average loss of {avg_loss}")

    print_tokens(
        evaluate(
            ckpt,
            start_index + len(prompt),
            length,
            temperature=temperature,
            top_k=top_k,
        )
    )


def model_info_operation(checkpoint_path: str, batch_size: int):
    print("Displaying model summary...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    summary(
        ckpt.model,
        input_data=torch.zeros(
            (batch_size, 64), dtype=torch.int64, device=ckpt.model.device
        ),
    )


def train_hypernet_operation(
    dataset_path: str,
    checkpoint_path: str,
    study_path: str,
    batch_size: int,
    num_epochs: int,
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

    print("Training hypernet model...")
    ckpt.model.train()
    avg_loss = train_hypernet(
        ckpt,
        dataset=get_cifar10_dataset(dataset_path, train=True),
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    print(f"Training complete with an average loss of {avg_loss}")

    ckpt.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    evaluate_hypernet_operation(
        dataset_path,
        checkpoint_path,
    )


def evaluate_hypernet_operation(
    dataset_path: str,
    checkpoint_path: str,
):
    print("Evaluating hypernet model...")
    ckpt = MLPCheckpoint.load(checkpoint_path)
    accuracy = evaluate_hypernet(
        ckpt,
        dataset=get_cifar10_dataset(dataset_path, train=False),
    )
    print(f"Test accuracy of the generated CNN: {accuracy:.2f}%")


def optimize_hypernet_operation(
    dataset_path: str,
    checkpoint_path: str,
    study_path: str,
    batch_size: int,
    num_epochs: int,
    trials: int,
):
    if Path(checkpoint_path).exists():
        print(f"Checkpoint file exists. Hyperparameters will not be optimized.")
        return

    print("Optimizing hyperparameters...")
    try:
        optimize_hyperparameters(
            study_path,
            get_cifar10_dataset(dataset_path, train=True),
            n_trials=trials,
            num_epochs=num_epochs,
            batch_size=batch_size,
            train_model="hypernet",
        )
    except KeyboardInterrupt:
        print("Optimization interrupted.")


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
        "--trials",
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
        help="Maximum length of the dataset",
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
        help="Maximum length of the dataset",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument(
        "--start", type=int, default=None, help="Starting index for evaluation"
    )
    evaluate_parser.add_argument(
        "--length", type=int, default=100, help="Number of characters to evaluate"
    )
    evaluate_parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    evaluate_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens to consider for sampling",
    )

    autocomplete_parser = subparsers.add_parser(
        "autocomplete", help="Autocomplete using the model"
    )
    autocomplete_parser.add_argument("prompt", help="Prompt string to autocomplete")
    autocomplete_parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Number of characters to predict after the prompt",
    )
    autocomplete_parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling"
    )
    autocomplete_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens to consider for sampling",
    )

    model_info_parser = subparsers.add_parser(
        "model_info", help="Display model summary"
    )
    model_info_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for the model summary"
    )

    train_hypernet = subparsers.add_parser(
        "train_hypernet", help="Test the hypernet model"
    )
    train_hypernet.add_argument("dataset_path", help="Path to the dataset")
    train_hypernet.add_argument(
        "--study-path",
        default="study.db",
        help="Path to the database for loading hyperparameters",
    )
    train_hypernet.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size"
    )
    train_hypernet.add_argument(
        "--num-epochs", type=int, default=2, help="Number of epochs"
    )

    evaluate_hypernet_parser = subparsers.add_parser(
        "evaluate_hypernet", help="Evaluate the hypernet model"
    )
    evaluate_hypernet_parser.add_argument("dataset_path", help="Path to the dataset")

    optimize_hypernet_parser = subparsers.add_parser(
        "optimize_hypernet", help="Optimize hyperparameters for hypernet"
    )
    optimize_hypernet_parser.add_argument("dataset_path", help="Path to the dataset")
    optimize_hypernet_parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter optimization",
    )
    optimize_hypernet_parser.add_argument(
        "--study-path",
        default="study.db",
        help="Path to the database for storing optimization studies",
    )
    optimize_hypernet_parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size"
    )
    optimize_hypernet_parser.add_argument(
        "--num-epochs", type=int, default=2, help="Number of epochs"
    )

    args = parser.parse_args()
    operations = {
        "optimize": lambda: optimize_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
            args.trials,
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
            args.checkpoint_path, args.start, args.length, args.temperature, args.top_k
        ),
        "autocomplete": lambda: autocomplete_operation(
            args.checkpoint_path, args.prompt, args.length, args.temperature, args.top_k
        ),
        "model_info": lambda: model_info_operation(
            args.checkpoint_path, args.batch_size
        ),
        "train_hypernet": lambda: train_hypernet_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
        ),
        "evaluate_hypernet": lambda: evaluate_hypernet_operation(
            args.dataset_path,
            args.checkpoint_path,
        ),
        "optimize_hypernet": lambda: optimize_hypernet_operation(
            args.dataset_path,
            args.checkpoint_path,
            args.study_path,
            args.batch_size,
            args.num_epochs,
            args.trials,
        ),
    }
    operations[args.operation]()
