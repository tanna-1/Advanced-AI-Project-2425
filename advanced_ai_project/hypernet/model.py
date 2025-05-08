import torch.nn as nn

from .utils import generate_param_info


class DynamicWeightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DynamicWeightCNN, self).__init__()
        self.num_classes = num_classes

        # Create temporary modules to get parameter sizes
        temp_modules = {
            "conv1": nn.Conv2d(3, 32, kernel_size=3, padding=1),
            "conv2": nn.Conv2d(32, 64, kernel_size=3, padding=1),
            "conv3": nn.Conv2d(64, 128, kernel_size=3, padding=1),
            "fc1": nn.Linear(128 * 4 * 4, 512),
            "fc2": nn.Linear(512, num_classes),
        }

        # Generate parameter information
        self.param_shapes, self.param_sizes, self.total_params = generate_param_info(
            temp_modules
        )

    def forward(self, x, weight_tensor):
        # Ensure weight tensor has enough elements
        if weight_tensor.numel() < self.total_params:
            raise ValueError(
                f"Weight tensor needs {self.total_params} elements but has {weight_tensor.numel()}"
            )

        # Extract weights while preserving gradient flow
        offset = 0

        # Conv1 weights and bias
        conv1_w = weight_tensor[offset : offset + self.param_sizes["conv1_w"]].reshape(
            self.param_shapes["conv1_w"]
        )
        offset += self.param_sizes["conv1_w"]
        conv1_b = weight_tensor[offset : offset + self.param_sizes["conv1_b"]]
        offset += self.param_sizes["conv1_b"]

        # Conv2 weights and bias
        conv2_w = weight_tensor[offset : offset + self.param_sizes["conv2_w"]].reshape(
            self.param_shapes["conv2_w"]
        )
        offset += self.param_sizes["conv2_w"]
        conv2_b = weight_tensor[offset : offset + self.param_sizes["conv2_b"]]
        offset += self.param_sizes["conv2_b"]

        # Conv3 weights and bias
        conv3_w = weight_tensor[offset : offset + self.param_sizes["conv3_w"]].reshape(
            self.param_shapes["conv3_w"]
        )
        offset += self.param_sizes["conv3_w"]
        conv3_b = weight_tensor[offset : offset + self.param_sizes["conv3_b"]]
        offset += self.param_sizes["conv3_b"]

        # FC1 weights and bias
        fc1_w = weight_tensor[offset : offset + self.param_sizes["fc1_w"]].reshape(
            self.param_shapes["fc1_w"]
        )
        offset += self.param_sizes["fc1_w"]
        fc1_b = weight_tensor[offset : offset + self.param_sizes["fc1_b"]]
        offset += self.param_sizes["fc1_b"]

        # FC2 weights and bias
        fc2_w = weight_tensor[offset : offset + self.param_sizes["fc2_w"]].reshape(
            self.param_shapes["fc2_w"]
        )
        offset += self.param_sizes["fc2_w"]
        fc2_b = weight_tensor[offset : offset + self.param_sizes["fc2_b"]]

        # Define forward pass with functional operations
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
