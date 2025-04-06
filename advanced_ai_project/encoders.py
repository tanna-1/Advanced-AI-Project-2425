import torch


def binary_encode_numbers(
    numbers: torch.Tensor, num_bits: int, dtype: torch.dtype = torch.float32
):
    arange = torch.arange(num_bits, device=numbers.device)
    return ((numbers.unsqueeze(-1) & (1 << arange)) > 0).to(dtype=dtype)
