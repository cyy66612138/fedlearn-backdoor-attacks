import torch.nn as nn
import torch.nn.functional as F

from . import register_model


_SIMPLE_CONFIG = {
    "28": [30, 50, 100],  # MNIST, FashionMNIST, FEMNIST
    "32": [32, 64, 512],  # CIFAR-10, CINIC-10
}


@register_model("simplecnn")
class SimpleCNN(nn.Module):
    def __init__(self, input_size=(3, 32, 32), num_classes=10):
        super().__init__()
        self.model_config = _SIMPLE_CONFIG[f"{input_size[1]}"]
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=self.model_config[0], kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.model_config[0], out_channels=self.model_config[1], kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flattened_size = self._get_flattened_size(input_size)
        self.fc1 = nn.Linear(self.flattened_size, self.model_config[2])
        self.fc2 = nn.Linear(self.model_config[2], num_classes)

    def _get_flattened_size(self, input_size):
        height, width = input_size[1], input_size[2]
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        height = (height - 2) // 2 + 1
        width = (width - 2) // 2 + 1
        height = (height - 3) // 1 + 1
        width = (width - 3) // 1 + 1
        height = (height - 2) // 2 + 1
        width = (width - 2) // 2 + 1
        return self.model_config[1] * height * width

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x