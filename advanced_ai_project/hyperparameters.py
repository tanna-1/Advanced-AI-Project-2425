from torch.utils.data import Dataset
from typing import Any
import torch
import optuna

from .model import MLPCheckpoint


def _optuna_objective_wrap(dataset: Dataset, num_epochs: int, batch_size: int):

    def _optuna_objective(trial: optuna.Trial) -> float:
        nonlocal dataset
        nonlocal num_epochs

        # Fix the random seed for optimization of hyperparameters
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        params = {
            "expansion_factor": trial.suggest_float(
                "expansion_factor", 1.0, 4.0, step=0.5
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "use_selu": trial.suggest_categorical("use_selu", [True, False]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3),
        }
        return MLPCheckpoint.new_from_hyperparams(params).train(
            dataset=dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            return_loss_over_n=100,
        )

    return _optuna_objective


def load_hyperparameters(db_path: str) -> dict[str, Any]:
    return optuna.load_study(
        storage=f"sqlite:///{db_path}", study_name="hyperparameters"
    ).best_params


def optimize_hyperparameters(
    db_path: str,
    dataset: Dataset,
    n_trials: int,
    num_epochs: int,
    batch_size: int,
):
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        study_name="hyperparameters",
        load_if_exists=True,
    )

    study.optimize(
        _optuna_objective_wrap(
            dataset=dataset, num_epochs=num_epochs, batch_size=batch_size
        ),
        n_trials=n_trials,
    )
