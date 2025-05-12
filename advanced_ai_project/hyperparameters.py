from torch.utils.data import Dataset
from typing import Any, Callable
import optuna

from .model import MLPCheckpoint


def _optuna_objective_wrap(
    dataset: Dataset,
    num_epochs: int,
    batch_size: int,
    train_function: Callable[..., float],
):
    """
    Create an Optuna objective function with fixed dataset and training parameters.

    Args:
        dataset (Dataset): Dataset to use for training.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        train_function: Function to use for training the model.

    Returns:
        callable: An Optuna objective function.
    """

    def _optuna_objective(trial: optuna.Trial) -> float:
        nonlocal dataset
        nonlocal num_epochs

        # Expansion factors lower than 1.0 were not studied in the paper. However, we'll allow the optimizer to test them.
        params = {
            "expansion_factor": trial.suggest_float(
                "expansion_factor", 0.5, 4.0, step=0.5
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_selu": trial.suggest_categorical("use_selu", [True, False]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3),
        }
        return train_function(
            MLPCheckpoint.new_from_hyperparams(params),
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            return_loss_over_n=100,
        )

    return _optuna_objective


def load_hyperparameters(db_path: str) -> dict[str, Any]:
    """
    Load the best hyperparameters from an Optuna study.

    Args:
        db_path (str): Path to the SQLite database containing the Optuna study.

    Returns:
        dict[str, Any]: Dictionary of best hyperparameters.
    """
    return optuna.load_study(
        storage=f"sqlite:///{db_path}", study_name="hyperparameters"
    ).best_params


def optimize_hyperparameters(
    db_path: str,
    dataset: Dataset,
    n_trials: int,
    num_epochs: int,
    batch_size: int,
    train_function: Callable[..., float],
):
    """
    Optimize hyperparameters using Optuna.

    Args:
        db_path (str): Path to store SQLite database with Optuna study.
        dataset (Dataset): Dataset to use for training during optimization.
        n_trials (int): Number of trials to run.
        num_epochs (int): Number of epochs per trial.
        batch_size (int): Batch size for training.
        train_function (Callable[..., float]): Function to use for training the model.
    """
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        study_name="hyperparameters",
        load_if_exists=True,
    )

    study.optimize(
        _optuna_objective_wrap(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            train_function=train_function,
        ),
        n_trials=n_trials,
    )

    print("Best hyperparameters: ", study.best_params)
