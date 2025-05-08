import torch
from torch.utils.data import DataLoader, Dataset

from .model import DynamicWeightCNN
from .utils import compute_weight_tensor
from ..model import MLPCheckpoint


def evaluate(ckpt: MLPCheckpoint, dataset: Dataset, batch_size: int = 128):
    device = ckpt.model.device
    cnn_model = DynamicWeightCNN(num_classes=10).to(device)

    ckpt.model.eval()
    cnn_model.eval()

    correct = 0
    total = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Evaluate the MLP model to get the weight vectors
            weight_tensor = compute_weight_tensor(ckpt.model, cnn_model.total_params)

            outputs = cnn_model(images, weight_tensor)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
