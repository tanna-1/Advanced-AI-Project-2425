import collections
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset

from ..model import MLPCheckpoint


def train(
    ckpt: MLPCheckpoint,
    dataset: Dataset,
    num_epochs: int,
    batch_size: int,
    return_loss_over_n: int = 100,
    show_progress: bool = True,
):
    """
    Train a text prediction model on the given dataset.
    
    Args:
        ckpt (MLPCheckpoint): The checkpoint containing the model to train.
        dataset (Dataset): The dataset to train on.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        return_loss_over_n (int, optional): Number of most recent batches to average loss over. Defaults to 100.
        show_progress (bool, optional): Whether to show progress bars. Defaults to True.
        
    Returns:
        float: The average loss over the last return_loss_over_n batches.
    """
    loss_fn = nn.CrossEntropyLoss()

    # Last N losses
    loss_history = collections.deque(maxlen=return_loss_over_n)

    # shuffle=True causes issues with lazy datasets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    progress = lambda x: tqdm(x) if show_progress else x
    for _ in progress(range(num_epochs)):
        for inputs, targets in progress(dataloader):
            inputs = inputs.to(ckpt.model.device)
            targets = targets.to(ckpt.model.device)

            ckpt.last_seen_index = inputs.max().item()

            result = ckpt.model(inputs)

            loss = loss_fn(result, targets)
            loss_history.append(loss.item())

            ckpt.optimizer.zero_grad()
            loss.backward()
            ckpt.optimizer.step()

    return sum(loss_history) / len(loss_history)
