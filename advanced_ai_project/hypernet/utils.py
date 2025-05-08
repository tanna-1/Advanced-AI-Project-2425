from math import ceil
import torch
import torch.nn as nn

from ..model import OUT_DIM, MLPModel


def generate_param_info(modules_dict: dict[str, nn.Module]):
    """
    Generate parameter shapes and sizes for a given dictionary of modules.

    Args:
        modules_dict (dict): Dictionary where keys are module names and values are nn.Module instances.

    Returns:
        tuple: A tuple containing:
            - param_shapes (dict): Dictionary of parameter shapes.
            - param_sizes (dict): Dictionary of parameter sizes.
            - total_params (int): Total number of parameters.
    """

    param_shapes = {}
    param_sizes = {}

    for prefix, module in modules_dict.items():
        param_shapes[f"{prefix}_w"] = module.weight.shape
        param_shapes[f"{prefix}_b"] = module.bias.shape

        param_sizes[f"{prefix}_w"] = module.weight.numel()  # type: ignore
        param_sizes[f"{prefix}_b"] = module.bias.numel()  # type: ignore

    total_params = sum(param_sizes.values())

    return param_shapes, param_sizes, total_params


def compute_weight_tensor(model: MLPModel, param_count: int) -> torch.Tensor:
    index_tensor = torch.arange(
        0,
        ceil(param_count / OUT_DIM),
        dtype=torch.int64,
        device=model.device,
    )
    weight_vectors = model(index_tensor)

    # Normalize the weight vectors
    weight_vectors = torch.tanh(weight_vectors)

    # Flatten the weight vectors
    return weight_vectors.flatten()
