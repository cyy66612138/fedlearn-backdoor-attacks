import torch.nn as nn

from . import register_model


@register_model("lr")
class LogisticRegression(nn.Module):
    def __init__(self, input_dim=32 * 32, num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.linear(x)


