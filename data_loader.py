"""
Unified Data Loader for FL Research Framework
Supports: Image, Time Series, Audio, and Text datasets
Memory-efficient loading with non-IID partitioning
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from PIL import Image
from download_datasets import ensure_dataset_available


class BaseDataset(ABC):
    """Base class for all datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.data_type = config.get('type', 'unknown')
        self.data_root = config.get('data_root', './data')
        
    @abstractmethod
    def get_train_data(self) -> Dataset:
        pass
    
    @abstractmethod
    def get_test_data(self) -> Dataset:
        pass
    
    @abstractmethod
    def get_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abstractmethod
    def get_data_shape(self) -> Tuple[int, ...]:
        pass
    
    def get_data_type(self) -> str:
        return self.data_type


class ImageDataset(BaseDataset):
    """Base class for image datasets"""
    
    def get_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get normalization values from config if available
        if 'mean' in self.config and 'std' in self.config:
            mean = torch.tensor(self.config['mean'])
            std = torch.tensor(self.config['std'])
        else:
            # Fallback to calculated values based on dataset name
            if self.name.lower() == 'cifar10':
                mean = torch.tensor([0.4914, 0.4822, 0.4465])
                std = torch.tensor([0.2470, 0.2435, 0.2616])
            elif self.name.lower() == 'cifar100':
                mean = torch.tensor([0.5071, 0.4865, 0.4409])
                std = torch.tensor([0.2673, 0.2564, 0.2762])
            elif self.name.lower() == 'gtsrb':
                mean = torch.tensor([0.3417, 0.3126, 0.3216])
                std = torch.tensor([0.2737, 0.2607, 0.2662])
            elif self.name.lower() == 'tinyimagenet':
                mean = torch.tensor([0.4802, 0.4481, 0.3975])
                std = torch.tensor([0.2764, 0.2689, 0.2816])
            elif self.name.lower() == 'svhn':
                # Commonly used SVHN statistics
                mean = torch.tensor([0.4377, 0.4438, 0.4728])
                std = torch.tensor([0.1980, 0.2010, 0.1970])
            elif self.name.lower() == 'mnist':
                mean = torch.tensor([0.1307])
                std = torch.tensor([0.3081])
            elif self.name.lower() == 'fashionmnist':
                mean = torch.tensor([0.2860])
                std = torch.tensor([0.3530])
            elif self.name.lower() == 'femnist':
                mean = torch.tensor([0.1722])
                std = torch.tensor([0.3309])
            else:
                # Default fallback
                mean = torch.tensor([0.5, 0.5, 0.5])
                std = torch.tensor([0.5, 0.5, 0.5])
        
        return mean, std
    
    def get_data_shape(self) -> Tuple[int, ...]:
        if self.name.lower() in ['mnist', 'fashionmnist', 'femnist']:
            return (1, 28, 28)  # Grayscale, 28x28
        elif self.name.lower() in ['cifar10', 'cifar100', 'gtsrb', 'svhn']:
            return (3, 32, 32)  # RGB, 32x32
        elif self.name.lower() == 'tinyimagenet':
            return (3, 64, 64)  # RGB, 64x64
        else:
            return (3, 32, 32)  # Default fallback


class CIFAR10Dataset(ImageDataset):
    """CIFAR-10 dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=transform
        )


class MNISTDataset(ImageDataset):
    """MNIST dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.MNIST(
            root=self.data_root, train=True, download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.MNIST(
            root=self.data_root, train=False, download=True, transform=transform
        )


class CIFAR100Dataset(ImageDataset):
    """CIFAR-100 dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.CIFAR100(
            root=self.data_root, train=True, download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=transform
        )


class FashionMNISTDataset(ImageDataset):
    """Fashion-MNIST dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.FashionMNIST(
            root=self.data_root, train=True, download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.FashionMNIST(
            root=self.data_root, train=False, download=True, transform=transform
        )


class GTSRBDataset(ImageDataset):
    """GTSRB (German Traffic Sign Recognition Benchmark) dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.GTSRB(
            root=self.data_root, split='train', download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.GTSRB(
            root=self.data_root, split='test', download=True, transform=transform
        )


class TinyImageNetSimpleDataset(Dataset):
    """Simple dataset wrapper for TinyImageNet samples"""
    
    def __init__(self, samples: List[str], targets: List[int], transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index"""
        try:
            image = Image.open(self.samples[idx]).convert('RGB')
            label = self.targets[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {self.samples[idx]}, error: {e}")
            # Return a dummy sample to avoid breaking the training loop
            dummy_image = torch.zeros(3, 64, 64)
            return dummy_image, 0

            
class TinyImageNetDataset(ImageDataset):
    """TinyImageNet dataset implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_root = config.get('data_root', './data')
        # New download logic stores TinyImageNet directly at data_root/tiny-imagenet-200
        # The data_root already includes the dataset subdirectory from ensure_dataset_available
        self.base_dir = self.data_root
        
        # Load class names once and reuse
        self._load_class_mapping()
    
    def _load_class_mapping(self) -> None:
        """Load class names and create mapping"""
        wnids_path = os.path.join(self.base_dir, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(f"Class names file not found: {wnids_path}")
        
        with open(wnids_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        print(f"🔍 Loaded {len(self.class_names)} classes for TinyImageNet")
    
    def get_data_shape(self) -> Tuple[int, ...]:
        """Return data shape for TinyImageNet"""
        return (3, 64, 64)  # RGB, 64x64
    
    def get_train_data(self) -> Dataset:
        """Get training dataset"""
        return self._build_dataset(split='train')

    def get_test_data(self) -> Dataset:
        """Get test/validation dataset"""
        return self._build_dataset(split='val')

    def _build_dataset(self, split: str) -> Dataset:
        """Build dataset for given split"""
        mean, std = self.get_normalization()
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.480, 0.448, 0.398],
        #                         std=[0.277, 0.269, 0.282])
        # ])

        samples, targets = self._load_samples(split)
        
        if not samples:
            raise ValueError(f"No samples found for split: {split}")
        
        print(f"✅ Loaded {len(samples)} samples for {split} split")
        return TinyImageNetSimpleDataset(samples, targets, transform)
    
    def _load_samples(self, split: str) -> Tuple[List[str], List[int]]:
        """Load samples and targets for given split"""
        samples = []
        targets = []
        
        if split == 'train':
            samples, targets = self._load_train_samples()
        elif split == 'val':
            samples, targets = self._load_val_samples()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        return samples, targets
    
    def _load_train_samples(self) -> Tuple[List[str], List[int]]:
        """Load training samples from class folders"""
        samples = []
        targets = []
        
        train_dir = os.path.join(self.base_dir, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        for class_name in self.class_names:
            img_dir = os.path.join(train_dir, class_name, 'images')
            if not os.path.isdir(img_dir):
                print(f"⚠️  Warning: Class directory not found: {img_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            image_files = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for fname in image_files:
                samples.append(os.path.join(img_dir, fname))
                targets.append(class_idx)
        
        return samples, targets
    
    def _load_val_samples(self) -> Tuple[List[str], List[int]]:
        """Load validation samples from annotations file"""
        samples = []
        targets = []
        
        val_dir = os.path.join(self.base_dir, 'val')
        img_dir = os.path.join(val_dir, 'images')
        ann_path = os.path.join(val_dir, 'val_annotations.txt')
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Validation images directory not found: {img_dir}")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Validation annotations file not found: {ann_path}")
        
        valid_samples = 0
        invalid_samples = 0
        
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    fname, class_name = parts[0], parts[1]
                    
                    if class_name in self.class_to_idx:
                        img_path = os.path.join(img_dir, fname)
                        if os.path.exists(img_path):
                            samples.append(img_path)
                            targets.append(self.class_to_idx[class_name])
                            valid_samples += 1
                        else:
                            invalid_samples += 1
                    else:
                        invalid_samples += 1
        
        if invalid_samples > 0:
            print(f"⚠️  Warning: {invalid_samples} invalid samples skipped")
        
        return samples, targets





class FEMNISTDataset(ImageDataset):
    """FEMNIST (Federated EMNIST) dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.EMNIST(
            root=self.data_root, split='letters', train=True, download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.EMNIST(
            root=self.data_root, split='letters', train=False, download=True, transform=transform
        )


class TimeSeriesDataset(BaseDataset):
    """Base class for time series datasets"""
    
    def get_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([0.0]), torch.tensor([1.0])
    
    def get_data_shape(self) -> Tuple[int, ...]:
        return (12, 358, 1)  # Default for PEMS-like data


class PEMS03Dataset(TimeSeriesDataset):
    """PEMS03 traffic dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        num_samples = 1000
        time_steps = self.config.get('sequence_length', 12)
        num_variables = self.config.get('num_variables', 358)
        
        # Create dummy time series data
        data = torch.randn(num_samples, time_steps, num_variables, 1)
        labels = torch.randint(0, 10, (num_samples,))
        
        return torch.utils.data.TensorDataset(data, labels)
    
    def get_test_data(self) -> Dataset:
        num_samples = 200
        time_steps = self.config.get('sequence_length', 12)
        num_variables = self.config.get('num_variables', 358)
        
        data = torch.randn(num_samples, time_steps, num_variables, 1)
        labels = torch.randint(0, 10, (num_samples,))
        
        return torch.utils.data.TensorDataset(data, labels)
    
    def get_data_shape(self) -> Tuple[int, ...]:
        time_steps = self.config.get('sequence_length', 12)
        num_variables = self.config.get('num_variables', 358)
        return (time_steps, num_variables, 1)


class AudioDataset(BaseDataset):
    """Base class for audio datasets"""
    
    def get_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([0.0]), torch.tensor([1.0])
    
    def get_data_shape(self) -> Tuple[int, ...]:
        return (16000,)  # Default audio length


class SpeechCommandsDataset(AudioDataset):
    """Speech Commands dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        num_samples = 2000
        audio_length = self.config.get('audio_length', 16000)
        
        # Create dummy audio data
        data = torch.randn(num_samples, audio_length)
        labels = torch.randint(0, 35, (num_samples,))
        
        return torch.utils.data.TensorDataset(data, labels)
    
    def get_test_data(self) -> Dataset:
        num_samples = 400
        audio_length = self.config.get('audio_length', 16000)
        
        data = torch.randn(num_samples, audio_length)
        labels = torch.randint(0, 35, (num_samples,))
        
        return torch.utils.data.TensorDataset(data, labels)
    
    def get_data_shape(self) -> Tuple[int, ...]:
        audio_length = self.config.get('audio_length', 16000)
        return (audio_length,)


class SVHNDataset(ImageDataset):
    """SVHN dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.SVHN(
            root=self.data_root, split='train', download=True, transform=transform
        )
    
    def get_test_data(self) -> Dataset:
        mean, std = self.get_normalization()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return torchvision.datasets.SVHN(
            root=self.data_root, split='test', download=True, transform=transform
        )


class TextDataset(BaseDataset):
    """Base class for text datasets"""
    
    def get_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return None, None
    
    def get_data_shape(self) -> Tuple[int, ...]:
        return (512,)  # Default sequence length


class IMDBDataset(TextDataset):
    """IMDB dataset implementation"""
    
    def get_train_data(self) -> Dataset:
        num_samples = 1000
        seq_length = self.config.get('sequence_length', 512)
        
        # Create dummy text data (tokenized)
        data = torch.randint(0, 10000, (num_samples, seq_length))
        labels = torch.randint(0, 2, (num_samples,))
        
        return torch.utils.data.TensorDataset(data, labels)
    
    def get_test_data(self) -> Dataset:
        num_samples = 200
        seq_length = self.config.get('sequence_length', 512)
        
        data = torch.randint(0, 10000, (num_samples, seq_length))
        labels = torch.randint(0, 2, (num_samples,))
        
        return torch.utils.data.TensorDataset(data, labels)
    
    def get_data_shape(self) -> Tuple[int, ...]:
        seq_length = self.config.get('sequence_length', 512)
        return (seq_length,)


class DataLoaderManager:
    """Unified data loader manager for all dataset types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset = self._create_dataset()
        self.train_dataset = None
        self.test_dataset = None
        self.client_datasets = []
        
    def _create_dataset(self) -> BaseDataset:
        """Create dataset instance based on config"""
        dataset_name = self.config['dataset']['name']
        dataset_type = self.config['dataset']['type']
        
        # Ensure dataset is available (download if necessary)
        # Returns the path where dataset is stored (data_root/dataset_subdirectory)
        base_data_root = self.config['dataset'].get('data_root', './data')
        dataset_path = ensure_dataset_available(dataset_name, base_data_root)
        
        # Update config with the actual dataset path (includes dataset subdirectory)
        dataset_config = self.config['dataset'].copy()
        dataset_config['data_root'] = dataset_path
        
        if dataset_type == 'image':
            if dataset_name.lower() == 'cifar10':
                return CIFAR10Dataset(dataset_config)
            elif dataset_name.lower() == 'cifar100':
                return CIFAR100Dataset(dataset_config)
            elif dataset_name.lower() == 'mnist':
                return MNISTDataset(dataset_config)
            elif dataset_name.lower() == 'fashionmnist':
                return FashionMNISTDataset(dataset_config)
            elif dataset_name.lower() == 'gtsrb':
                return GTSRBDataset(dataset_config)
            elif dataset_name.lower() == 'tinyimagenet':
                return TinyImageNetDataset(dataset_config)
            elif dataset_name.lower() == 'femnist':
                return FEMNISTDataset(dataset_config)
            elif dataset_name.lower() == 'svhn':
                return SVHNDataset(dataset_config)
            else:
                raise ValueError(f"Unknown image dataset: {dataset_name}")
        
        elif dataset_type == 'time_series':
            if dataset_name.lower() == 'pems03':
                return PEMS03Dataset(self.config['dataset'])
            else:
                raise ValueError(f"Unknown time series dataset: {dataset_name}")
        
        elif dataset_type == 'audio':
            if dataset_name.lower() == 'speech_commands':
                return SpeechCommandsDataset(self.config['dataset'])
            else:
                raise ValueError(f"Unknown audio dataset: {dataset_name}")
        
        elif dataset_type == 'text':
            if dataset_name.lower() == 'imdb':
                return IMDBDataset(self.config['dataset'])
            else:
                raise ValueError(f"Unknown text dataset: {dataset_name}")
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def load_datasets(self) -> Tuple[Dataset, Dataset, List[Subset]]:
        """Load train, test, and client datasets"""
        print(f"📊 Loading {self.config['dataset']['name']} dataset...")
        
        # Load train and test datasets
        self.train_dataset = self.dataset.get_train_data()
        self.test_dataset = self.dataset.get_test_data()
        
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Test samples: {len(self.test_dataset)}")
        
        # Create client datasets with non-IID partitioning
        self.client_datasets = self._create_non_iid_partitions(
            self.train_dataset,
            num_clients=self.config['federated_learning']['num_clients'],
            alpha=self.config['dataset'].get('alpha', 0.5)
        )
        
        # print(f"   Client datasets: {len(self.client_datasets)}")
        # for i, client_dataset in enumerate(self.client_datasets):
        #     print(f"     Client {i}: {len(client_dataset)} samples")
        
        return self.train_dataset, self.test_dataset, self.client_datasets
    
    def _create_non_iid_partitions(self, dataset: Dataset, num_clients: int, alpha: float) -> List[Subset]:
        """Create non-IID data partitions using Dirichlet distribution"""
        print(f"📊 Creating non-IID partitions for {num_clients} clients...")
    
        
        # Get class labels
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        else:
            # For TensorDataset, extract labels from the second element
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
        num_classes = len(set(targets))
        
        
        # Create Dirichlet distribution
        partitions = []
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            np.random.shuffle(class_indices)
            
            # Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.cumsum(proportions)
            proportions[-1] = 1.0
            
            # Split class data among clients
            class_partitions = np.split(class_indices, (proportions[:-1] * len(class_indices)).astype(int))
            
            if len(partitions) == 0:
                partitions = [[] for _ in range(num_clients)]
            
            for client_id, client_indices in enumerate(class_partitions):
                partitions[client_id].extend(client_indices)
        
        # # # Print label distribution for each client
        # for idx, client_indices in enumerate(partitions):
        #     labels, counts = np.unique(targets[client_indices], return_counts=True)
        #     label_count_dict = dict(zip(labels.tolist(), counts.tolist()))
        #     print(f"Client {idx} with total {len(client_indices)} samples, label counts: {label_count_dict}")


        # # After creating partitions, print first 10 indices for each client
        # for client_id, client_indices in enumerate(partitions):
        #     if len(client_indices) > 0:
        #         first_10_indices = client_indices[:10]
        #         print(f"📊 Client {client_id}: First 10 indices: {first_10_indices}")
            
        client_datasets = []
        for client_indices in partitions:
            if len(client_indices) > 0:
                client_datasets.append(Subset(dataset, client_indices))
            else:
                # Empty dataset fallback
                client_datasets.append(Subset(dataset, [0]))
        
        print(f"✅ Created {len(client_datasets)} client datasets")
        return client_datasets
    
    def get_test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get test data loader"""
        if batch_size is None:
            batch_size = self.config['evaluation']['batch_size']
        
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    
    def get_client_loader(self, client_id: int, batch_size: Optional[int] = None) -> DataLoader:
        """Get client data loader"""
        if batch_size is None:
            batch_size = self.config['federated_learning']['batch_size']
        
        return DataLoader(
            self.client_datasets[client_id],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': self.dataset.name,
            'type': self.dataset.get_data_type(),
            'data_shape': self.dataset.get_data_shape(),
            'num_classes': self._get_num_classes(),
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'num_clients': len(self.client_datasets),
            'client_samples': [len(client_dataset) for client_dataset in self.client_datasets]
        }
    
    def _get_num_classes(self) -> int:
        """Get number of classes in the dataset"""
        if hasattr(self.dataset, 'classes'):
            return len(self.dataset.classes)
        else:
            # Try to get from dataset config first, then model config
            dataset_config = self.config.get('dataset', {})
            if 'num_classes' in dataset_config:
                return dataset_config['num_classes']
            else:
                # Fallback to model config
                return self.config['model'].get('num_classes', 10)
    
    def create_poisoned_dataset(self, client_id: int, attack_config: Dict) -> Subset:
        """Create poisoned dataset for a specific client"""
        # This would be implemented based on the specific attack
        # For now, return the original client dataset
        return self.client_datasets[client_id]
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get detailed data statistics"""
        if not self.client_datasets:
            return {}
        
        # Calculate class distribution for each client
        client_class_distributions = []
        
        for client_dataset in self.client_datasets:
            if hasattr(client_dataset.dataset, 'targets'):
                # For datasets with targets attribute
                client_targets = [client_dataset.dataset.targets[i] for i in client_dataset.indices]
            else:
                # For TensorDataset
                client_targets = [client_dataset[i][1] for i in range(len(client_dataset))]
            
            class_counts = np.bincount(client_targets)
            client_class_distributions.append(class_counts.tolist())
        
        return {
            'client_class_distributions': client_class_distributions,
            'total_classes': len(set([target for dist in client_class_distributions for target in range(len(dist)) if dist[target] > 0])),
            'min_samples_per_client': min([len(client_dataset) for client_dataset in self.client_datasets]),
            'max_samples_per_client': max([len(client_dataset) for client_dataset in self.client_datasets]),
            'avg_samples_per_client': np.mean([len(client_dataset) for client_dataset in self.client_datasets])
        }


# Dataset factory function
def create_dataset(dataset_config: Dict[str, Any]) -> BaseDataset:
    """Factory function to create dataset instances"""
    dataset_name = dataset_config['name']
    dataset_type = dataset_config.get('type', 'image')
    
    if dataset_type == 'image':
        if dataset_name.lower() == 'cifar10':
            return CIFAR10Dataset(dataset_config)
        elif dataset_name.lower() == 'cifar100':
            return CIFAR100Dataset(dataset_config)
        elif dataset_name.lower() == 'mnist':
            return MNISTDataset(dataset_config)
        elif dataset_name.lower() == 'fashionmnist':
            return FashionMNISTDataset(dataset_config)
        elif dataset_name.lower() == 'gtsrb':
            return GTSRBDataset(dataset_config)
        elif dataset_name.lower() == 'tinyimagenet':
            return TinyImageNetDataset(dataset_config)
        elif dataset_name.lower() == 'femnist':
            return FEMNISTDataset(dataset_config)
        elif dataset_name.lower() == 'svhn':
            return SVHNDataset(dataset_config)
        else:
            raise ValueError(f"Unknown image dataset: {dataset_name}")
    
    elif dataset_type == 'time_series':
        if dataset_name.lower() == 'pems03':
            return PEMS03Dataset(dataset_config)
        else:
            raise ValueError(f"Unknown time series dataset: {dataset_name}")
    
    elif dataset_type == 'audio':
        if dataset_name.lower() == 'speech_commands':
            return SpeechCommandsDataset(dataset_config)
        else:
            raise ValueError(f"Unknown audio dataset: {dataset_name}")
    
    elif dataset_type == 'text':
        if dataset_name.lower() == 'imdb':
            return IMDBDataset(dataset_config)
        else:
            raise ValueError(f"Unknown text dataset: {dataset_name}")
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
