import torch


def _get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = _get_torch_device()
