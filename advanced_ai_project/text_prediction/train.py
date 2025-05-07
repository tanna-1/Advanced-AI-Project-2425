import collections
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset

from ..model import MLPCheckpoint


# Returns the average loss over the last N batches
def train(
    ckpt: MLPCheckpoint,
    dataset: Dataset,
    num_epochs: int,
    batch_size: int,
    return_loss_over_n: int = 100,
    show_progress: bool = True,
):
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
