import torch
import torch.nn as nn

from .utils import generate_weight_info


class DynamicWeightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DynamicWeightCNN, self).__init__()
        self.num_classes = num_classes

        # Generate weight information
        self.param_shapes, self.param_sizes, self.total_params = generate_weight_info(
            {
                "conv1": nn.Conv2d(3, 32, kernel_size=3, padding=1),
                "conv2": nn.Conv2d(32, 64, kernel_size=3, padding=1),
                "conv3": nn.Conv2d(64, 128, kernel_size=3, padding=1),
                "fc1": nn.Linear(128 * 4 * 4, 512),
                "fc2": nn.Linear(512, num_classes),
            }
        )

    def __extract_weight(
        self, weight_tensor: torch.Tensor, offset: int, name: str
    ) -> tuple[torch.Tensor, int]:
        w = weight_tensor[offset : offset + self.param_sizes[name]].reshape(
            self.param_shapes[name]
        )
        offset += self.param_sizes[name]
        return w, offset

    def forward(self, x, weight_tensor):
        offset = 0

        # Extract weights
        conv1_w, offset = self.__extract_weight(weight_tensor, offset, "conv1_w")
        conv2_w, offset = self.__extract_weight(weight_tensor, offset, "conv2_w")
        conv3_w, offset = self.__extract_weight(weight_tensor, offset, "conv3_w")
        fc1_w, offset = self.__extract_weight(weight_tensor, offset, "fc1_w")
        fc2_w, offset = self.__extract_weight(weight_tensor, offset, "fc2_w")

        # Extract biases
        conv1_b, offset = self.__extract_weight(weight_tensor, offset, "conv1_b")
        conv2_b, offset = self.__extract_weight(weight_tensor, offset, "conv2_b")
        conv3_b, offset = self.__extract_weight(weight_tensor, offset, "conv3_b")
        fc1_b, offset = self.__extract_weight(weight_tensor, offset, "fc1_b")
        fc2_b, offset = self.__extract_weight(weight_tensor, offset, "fc2_b")

        # First conv block
        x = nn.functional.conv2d(x, conv1_w, bias=conv1_b, padding=1)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # Second conv block
        x = nn.functional.conv2d(x, conv2_w, bias=conv2_b, padding=1)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # Third conv block
        x = nn.functional.conv2d(x, conv3_w, bias=conv3_b, padding=1)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = nn.functional.linear(x, fc1_w, bias=fc1_b)
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, fc2_w, bias=fc2_b)

        return x
