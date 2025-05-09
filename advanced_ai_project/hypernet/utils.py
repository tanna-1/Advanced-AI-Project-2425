from math import ceil
import torch
import torch.nn as nn

from ..model import OUT_DIM, MLPModel

"""
If the output of the hypernet is too far from the distribution of initialized CNN weights
the model is unable to converge, as it leads to exploding gradients within the CNN.

The problem likely could've been solved by implementing the initialization method in the paper
"Chang, O., Flokas, L., & Lipson, H. (2023). Principled weight initialization for hypernetworks. arXiv. https://doi.org/10.48550/arXiv.2312.08399".

However I opted for a simpler solution where the hypernet output is scaled
by a constant value to match the distribution of the CNN weights.

Measured distribution of the hypernet output:
    - Mean: 0.005435472819954157
    - Std: 0.6838817596435547
    - Max: 3.3123366832733154
    - Min: -3.272418975830078

Measured distribution of the CNN weights:
    - Mean: -0.0027166458312422037
    - Std: 0.03464247286319733
    - Max: 0.28350767493247986
    - Min: -0.29618480801582336

The value is the ratio of the standard deviations of the two distributions.
"""
HYPERNET_OUTPUT_SCALE = 0.034 / 0.68


def generate_weight_info(modules_dict: dict[str, nn.Module]):
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
    """
    Compute a weight tensor from the model output.

    Args:
        model (MLPModel): The MLP model to generate weights from.
        param_count (int): The total number of parameters needed.

    Returns:
        torch.Tensor: A flattened tensor of normalized weights.
    """
    index_tensor = torch.arange(
        0,
        ceil(param_count / OUT_DIM),
        dtype=torch.int64,
        device=model.device,
    )
    weight_tensor = model(index_tensor)

    # Scale the weights so that they have the correct variance
    weight_tensor = weight_tensor * HYPERNET_OUTPUT_SCALE

    return weight_tensor.flatten()
