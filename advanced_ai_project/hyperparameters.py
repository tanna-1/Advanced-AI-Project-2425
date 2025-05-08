from torch.utils.data import Dataset
from typing import Any, Literal
import optuna

from .model import MLPCheckpoint
from .text_prediction.train import train as train_text_prediction
from .hypernet.train import train as train_hypernet


def _optuna_objective_wrap(
    dataset: Dataset,
    num_epochs: int,
    batch_size: int,
    train_function,
):

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
            "hidden_depth": trial.suggest_int("hidden_depth", 4, 16, step=2),
            "hidden_width": trial.suggest_int("hidden_width", 128, 512, step=32),
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
    return optuna.load_study(
        storage=f"sqlite:///{db_path}", study_name="hyperparameters"
    ).best_params


def optimize_hyperparameters(
    db_path: str,
    dataset: Dataset,
    n_trials: int,
    num_epochs: int,
    batch_size: int,
    train_model: Literal["text_prediction", "hypernet"],
):
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        study_name="hyperparameters",
        load_if_exists=True,
    )

    train_function = (
        train_hypernet if train_model == "hypernet" else train_text_prediction
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
