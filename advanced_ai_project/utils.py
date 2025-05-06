import torch


def sample_top_k(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    assert k > 0, "k must be greater than 0"
    assert k <= logits.size(0), "k must be less than or equal to the number of logits"

    # Scale the logits by temperature
    logits = logits / temperature

    # Get the top-k indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = torch.softmax(top_k_logits, dim=0)

    # Sample from the distribution
    sampled_idx = int(torch.multinomial(probs, 1).item())
    return top_k_indices[sampled_idx]


def _get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = _get_torch_device()
