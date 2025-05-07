from torch.utils.data import Dataset
from typing import Any
import optuna

from .text_prediction.train import train
from .model import MLPCheckpoint


def _optuna_objective_wrap(dataset: Dataset, num_epochs: int, batch_size: int):

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
        return train(
            MLPCheckpoint.new_from_hyperparams(params),
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

    print("Best hyperparameters: ", study.best_params)
