import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def sample_top_k(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """
    Sample from the top-k logits of a distribution.
    
    Args:
        logits (torch.Tensor): The logits to sample from.
        k (int): Number of top logits to consider.
        temperature (float): Temperature for scaling logits.
        
    Returns:
        torch.Tensor: The sampled token index.
    """
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


def get_cifar10_dataset(data_root: str, train: bool = True) -> datasets.CIFAR10:
    """
    Create a CIFAR-10 dataset with standard transforms.
    
    Args:
        data_root (str): Root directory to store the dataset.
        train (bool, optional): Whether to load the training set. Defaults to True.
        
    Returns:
        datasets.CIFAR10: The CIFAR-10 dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return datasets.CIFAR10(
        root=data_root, train=train, download=True, transform=transform
    )


def _get_torch_device() -> torch.device:
    """
    Get the best available torch device.
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = _get_torch_device()
