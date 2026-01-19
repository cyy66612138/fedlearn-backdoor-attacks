"""
Defense implementations for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from abc import ABC, abstractmethod


class BaseDefense(ABC):
    """Base class for all defenses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.start_round = config.get('start_round', 0)
        self.frequency = config.get('frequency', 1)
    
    @abstractmethod
    def apply(self, global_model: nn.Module, client_models: List[nn.Module], 
              round_num: int, **kwargs) -> nn.Module:
        pass
    
    def should_apply(self, round_num: int) -> bool:
        return round_num >= self.start_round and (round_num - self.start_round) % self.frequency == 0


class LocalDefense(BaseDefense):
    """Local defense for clients"""
    
    def apply(self, model: nn.Module, round_num: int) -> nn.Module:
        # Simple local defense - could be gradient clipping, etc.
        return model


class AnomalyDetectionDefense(BaseDefense):
    """Anomaly detection defense for server"""
    
    def apply(self, global_model: nn.Module, client_models: List[nn.Module], 
              round_num: int, **kwargs) -> List[nn.Module]:
        # Filter out anomalous clients
        threshold = self.config.get('threshold', 0.95)
        normal_clients = []
        
        for client_model in client_models:
            similarity = self._compute_similarity(global_model, client_model)
            if similarity > threshold:
                normal_clients.append(client_model)
        
        return normal_clients if normal_clients else client_models
    
    def _compute_similarity(self, model1: nn.Module, model2: nn.Module) -> float:
        """Compute similarity between two models"""
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        similarities = []
        for name in state1:
            if name in state2:
                tensor1 = state1[name]
                tensor2 = state2[name]
                # Ensure floating dtype for cosine similarity and non-empty tensors
                if tensor1.numel() == 0 or tensor2.numel() == 0:
                    continue
                vec1 = tensor1.flatten().float()
                vec2 = tensor2.flatten().float()
                cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
                similarities.append(cos_sim.item())
        
        return np.mean(similarities) if similarities else 0.0


def create_defense(defense_config: Dict[str, Any]) -> BaseDefense:
    """Factory function to create defense instances"""
    defense_name = defense_config['name']
    
    if defense_name == 'LocalDefense':
        return LocalDefense(defense_config)
    elif defense_name == 'AnomalyDetection':
        return AnomalyDetectionDefense(defense_config)
    else:
        raise ValueError(f"Unknown defense: {defense_name}")
