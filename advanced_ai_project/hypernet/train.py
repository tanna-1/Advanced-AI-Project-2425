import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import deque

from .model import DynamicWeightCNN
from .utils import compute_weight_tensor
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
    Train a hypernetwork model on the given dataset.
    
    Args:
        ckpt (MLPCheckpoint): The checkpoint containing the MLP model to train.
        dataset (Dataset): The dataset to train on.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        return_loss_over_n (int, optional): Number of most recent batches to average loss over. Defaults to 100.
        show_progress (bool, optional): Whether to show progress bars. Defaults to True.
        
    Returns:
        float: The average loss over the last return_loss_over_n batches.
    """
    device = ckpt.model.device
    loss_fn = nn.CrossEntropyLoss()

    cnn_model = DynamicWeightCNN(num_classes=10).to(device)
    print(f"DynamicWeightCNN total params: {cnn_model.total_params}")

    loss_history = deque(maxlen=return_loss_over_n)
    ckpt.model.train()
    cnn_model.train()

    progress_fn = lambda x: tqdm(x) if show_progress else x

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in progress_fn(range(num_epochs)):
        for inputs, labels in progress_fn(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            ckpt.optimizer.zero_grad()

            # Evaluate the MLP model to get the weights
            weight_tensor = compute_weight_tensor(ckpt.model, cnn_model.total_params)

            # Forward pass through the CNN model using the generated weights
            outputs = cnn_model(inputs, weight_tensor)

            # Calculate the loss of the CNN model
            loss = loss_fn(outputs, labels)
            loss_history.append(loss.item())

            # This should optimize the MLP model to predict better weights for the CNN
            loss.backward()
            ckpt.optimizer.step()

    return sum(loss_history) / len(loss_history)
