"""
Model factory and model definitions
"""

import torch.nn as nn
from typing import Dict, Any

def create_model_from_dataset_config(config: Dict[str, Any]) -> nn.Module:
    """Create model with automatic parameter detection from dataset config"""
    # TODO: remove later for visualization purpose
    model_name = config['model']['name']
    data_shape = config['dataset']['data_shape'] # (image_size, num_channels)
    num_channels, input_dim = data_shape[0], data_shape[1]
    num_classes = config['dataset']['num_classes']
    # mnist: (1, 28, 28), cifar10: (3, 32, 32), tinyimagenet: (3, 64, 64)

    if model_name in ['resnet18', 'simplecnn']:
        from core.custom_models import get_model as get_custom_model
        # print(f"Creating custom model: {model_name} with input_dim: {num_channels}, num_channels: {num_classes}, input_dim: {input_dim}")
        return get_custom_model(
            model_name=model_name,
            num_channels=num_channels,
            num_classes=num_classes,
            num_dims=input_dim,
            device=None,
        )