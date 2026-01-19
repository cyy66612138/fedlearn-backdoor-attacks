"""
Edge-case dataset loaders for Edge-case Backdoor Attack
Adapted from FLPoison's edge_dataset.py
"""

import os
import pickle
import numpy as np
import torch
from torchvision import datasets
from typing import Tuple, Optional


class ARDISLoader:
    """
    ARDIS: Swedish historical handwritten digit dataset
    Source: https://ardisdataset.github.io/ARDIS/
    
    In edge-case backdoor, we use label 7 images → target_label
    """
    
    def __init__(self, target_label: int = 1, root: str = "./data"):
        self.root = root
        self.source_label = 7  # ARDIS label 7 images
        self.target_label = target_label
        self.data_path = os.path.join(root, 'ARDIS')
        self.filenames = [
            'ARDIS_train_2828.csv', 'ARDIS_train_labels.csv',
            'ARDIS_test_2828.csv', 'ARDIS_test_labels.csv'
        ]
        
        if self.source_label == self.target_label:
            raise ValueError(f"Source label ({self.source_label}) and target label ({self.target_label}) must be different")
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load ARDIS dataset from CSV files"""
        all_files_exist = all(os.path.exists(os.path.join(self.data_path, file))
                              for file in self.filenames)
        
        if not all_files_exist:
            raise FileNotFoundError(
                f"ARDIS dataset files not found in {self.data_path}. "
                f"Please download ARDIS dataset from https://ardisdataset.github.io/ARDIS/ "
                f"and extract to {self.data_path}/"
            )
        
        # Load CSV files
        def load_csv(idx):
            filepath = os.path.join(self.data_path, self.filenames[idx])
            return torch.from_numpy(np.loadtxt(filepath, dtype='float32'))
        
        train_images, train_labels = load_csv(0), load_csv(1)
        test_images, test_labels = load_csv(2), load_csv(3)
        
        # Reshape to [samples][width][height] (28x28 for MNIST)
        def to_mnist_shape(x):
            return x.reshape(x.shape[0], 28, 28)
        
        train_images = to_mnist_shape(train_images)
        test_images = to_mnist_shape(test_images)
        
        # Convert one-hot encoded labels to integer labels
        def onehot_to_label(y):
            return torch.argmax(y, dim=1) if y.dim() > 1 else y
        
        train_labels = onehot_to_label(train_labels)
        test_labels = onehot_to_label(test_labels)
        
        # Filter to source label (7) images
        train_mask = (train_labels == self.source_label)
        self.train_images = train_images[train_mask]
        self.train_labels = torch.tensor([self.target_label] * len(self.train_images))
        
        self.test_images = test_images
        self.test_labels = torch.tensor([self.target_label] * len(test_labels))
    
    def get_train_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training edge-case samples"""
        return self.train_images, self.train_labels
    
    def get_test_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test edge-case samples"""
        return self.test_images, self.test_labels


class SouthwestAirlineLoader:
    """
    SouthwestAirline dataset for CIFAR10 edge-case backdoor
    Source: https://github.com/ksreenivasan/OOD_Federated_Learning
    
    Airplane images (label 0) → target_label
    """
    
    def __init__(self, target_label: int = 9, root: str = "./data/southwest"):
        self.root = root
        self.source_label = 0  # Airplane label in CIFAR10
        self.target_label = target_label
        self.filenames = [
            'southwest_images_new_train.pkl',
            'southwest_images_new_test.pkl'
        ]
        
        if self.source_label == self.target_label:
            raise ValueError(f"Source label ({self.source_label}) and target label ({self.target_label}) must be different")
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load SouthwestAirline dataset from pickle files"""
        all_files_exist = all(os.path.exists(os.path.join(self.root, file))
                              for file in self.filenames)
        
        if not all_files_exist:
            raise FileNotFoundError(
                f"SouthwestAirline dataset files not found in {self.root}. "
                f"Please download from https://github.com/ksreenivasan/OOD_Federated_Learning/tree/master/saved_datasets "
                f"and place pickle files in {self.root}/"
            )
        
        # Load pickle files
        with open(os.path.join(self.root, self.filenames[0]), 'rb') as f:
            self.train_images = pickle.load(f)
        
        with open(os.path.join(self.root, self.filenames[1]), 'rb') as f:
            self.test_images = pickle.load(f)
        
        # Convert to tensors if needed
        if isinstance(self.train_images, np.ndarray):
            self.train_images = torch.from_numpy(self.train_images).float()
        if isinstance(self.test_images, np.ndarray):
            self.test_images = torch.from_numpy(self.test_images).float()
        
        # Create labels (all target_label)
        self.train_labels = torch.tensor([self.target_label] * len(self.train_images))
        self.test_labels = torch.tensor([self.target_label] * len(self.test_images))
    
    def get_train_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training edge-case samples"""
        return self.train_images, self.train_labels
    
    def get_test_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test edge-case samples"""
        return self.test_images, self.test_labels


def load_edge_case_dataset(dataset_name: str, target_label: int, data_root: str = "./data") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load edge-case dataset samples
    
    Args:
        dataset_name: 'mnist', 'cifar10', etc.
        target_label: Target label for edge-case samples
        data_root: Root directory for datasets
    
    Returns:
        Tuple of (images, labels) tensors
    """
    dataset_name = dataset_name.lower()
    
    if 'mnist' in dataset_name:
        loader = ARDISLoader(target_label=target_label, root=data_root)
        return loader.get_train_samples()
    elif dataset_name == 'cifar10':
        loader = SouthwestAirlineLoader(target_label=target_label, root=os.path.join(data_root, 'southwest'))
        return loader.get_train_samples()
    else:
        raise ValueError(f"Unsupported dataset for edge-case attack: {dataset_name}. "
                        f"Supported: 'mnist', 'cifar10'")

