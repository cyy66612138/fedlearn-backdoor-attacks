"""
Utility functions for debugging and helper operations
"""

import torch
from typing import Dict, Any


def print_model_weight_debug(model, label="Model", n=4):
    """Debug function to print model weights before and after aggregation"""
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                w = p.flatten()
                print(f"🔍 {label}: {w[:n].cpu().numpy()} | Sum: {w.sum().item():.6f}")
                break


def create_model_from_state_dict(model_state, model_config):
    """Create a model instance from state_dict for debugging"""
    if model_config['name'] == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=model_config['num_classes'])
        model.load_state_dict(model_state)
        return model
    else:
        return None