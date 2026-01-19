import os
import socket
import urllib.request
import zipfile
from torchvision import datasets

# Set socket timeout to 60 seconds for large downloads
socket.setdefaulttimeout(60)

# Base directory for all datasets (used when running script directly)
BASE_ROOT = "./data-test"

# Unified dataset configuration
# Format: lowercase_name -> (display_name, DatasetClass, subdirectory, has_train_test_split)
DATASET_CONFIG = {
    'cifar10': ('CIFAR10', datasets.CIFAR10, 'cifar10', True),
    'cifar100': ('CIFAR100', datasets.CIFAR100, 'cifar100', True),
    'mnist': ('MNIST', datasets.MNIST, 'mnist', True),
    'fashionmnist': ('FashionMNIST', datasets.FashionMNIST, 'fashion_mnist', True),
    'svhn': ('SVHN', datasets.SVHN, 'svhn', False),
    'gtsrb': ('GTSRB', datasets.GTSRB, 'gtsrb', False),
    'tinyimagenet': ('TinyImageNet', None, 'tiny-imagenet-200', False),  # Custom download
}


def _download_torchvision_dataset(dataset_name: str, dataset_class, root_dir: str, 
                                   has_train_test_split: bool, data_root: str = None, verbose: bool = False):
    """
    Download a dataset from torchvision.
    
    Args:
        dataset_name: Human-readable name of the dataset
        dataset_class: The dataset class from torchvision.datasets
        root_dir: Subdirectory where the dataset will be stored
        has_train_test_split: Whether the dataset has train/test splits
        data_root: Base directory (if None, uses BASE_ROOT)
        verbose: Whether to print progress messages
    """
    if data_root is None:
        data_root = BASE_ROOT
    
    full_root = os.path.join(data_root, root_dir)
    os.makedirs(full_root, exist_ok=True)
    
    try:
        if has_train_test_split:
            if verbose:
                print(f"Downloading {dataset_name} training set...")
            dataset_class(root=full_root, train=True, download=True)
            if verbose:
                print(f"{dataset_name} training set downloaded.")
            
            if verbose:
                print(f"Downloading {dataset_name} test set...")
            dataset_class(root=full_root, train=False, download=True)
            if verbose:
                print(f"{dataset_name} test set downloaded.")
        else:
            # For datasets without train/test split (SVHN, GTSRB)
            if verbose:
                print(f"Downloading {dataset_name} train split...")
            dataset_class(root=full_root, split='train', download=True)
            if verbose:
                print(f"{dataset_name} train split downloaded.")
            
            if verbose:
                print(f"Downloading {dataset_name} test split...")
            dataset_class(root=full_root, split='test', download=True)
            if verbose:
                print(f"{dataset_name} test split downloaded.")
        
        return True
        
    except socket.timeout as e:
        print(f"✗ Timeout error downloading {dataset_name}: {e}")
        raise
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        raise


def _download_tinyimagenet(root_dir: str, data_root: str = None):
    """Download TinyImageNet dataset from Stanford CS231n."""
    if data_root is None:
        data_root = BASE_ROOT
    
    full_root = os.path.join(data_root, root_dir)
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    try:
        # Check if already extracted
        if os.path.exists(full_root) and os.path.isdir(full_root):
            if os.path.exists(os.path.join(full_root, "train")) and \
               os.path.exists(os.path.join(full_root, "val")):
                return True
        
        # Download zip file if not exists
        if not os.path.exists(zip_path):
            print(f"Downloading TinyImageNet zip file from {url}...")
            print("This may take several minutes (~237 MB)...")
            
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="", flush=True)
            
            urllib.request.urlretrieve(url, zip_path, reporthook=show_progress)
            print()
        
        # Extract zip file
        if not os.path.exists(full_root) or not os.path.exists(os.path.join(full_root, "train")):
            print(f"Extracting TinyImageNet to {full_root}...")
            os.makedirs(data_root, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_root)
        
        # Verify extraction
        train_dir = os.path.join(full_root, "train")
        val_dir = os.path.join(full_root, "val")
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            return True
        else:
            raise RuntimeError(f"Extraction verification failed. Expected directories not found in {full_root}")
        
    except socket.timeout as e:
        print(f"✗ Timeout error downloading TinyImageNet: {e}")
        raise
    except Exception as e:
        print(f"✗ Error downloading TinyImageNet: {e}")
        raise


def _dataset_exists(dataset_name: str, dataset_path: str) -> bool:
    """Check if dataset already exists"""
    if not os.path.exists(dataset_path):
        return False
    
    if dataset_name.lower() == 'tinyimagenet':
        return os.path.exists(os.path.join(dataset_path, 'train')) and \
               os.path.exists(os.path.join(dataset_path, 'val'))
    else:
        # For torchvision datasets, check if the subdirectory exists and has content
        return os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0


def ensure_dataset_available(dataset_name: str, data_root: str = "./data") -> str:
    """
    Ensure dataset is available, download if necessary.
    Can be used to download individual datasets programmatically.
    
    Args:
        dataset_name: Name of the dataset (lowercase, e.g., 'cifar10', 'mnist')
        data_root: Base directory where datasets are stored (default: './data')
    
    Returns:
        Path where dataset is stored (data_root/dataset_subdirectory)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIG.keys())}")
    
    display_name, dataset_class, subdirectory, has_train_test_split = DATASET_CONFIG[dataset_name]
    dataset_path = os.path.join(data_root, subdirectory)
    
    # Check if dataset already exists
    if _dataset_exists(dataset_name, dataset_path):
        return dataset_path
    
    print(f"📥 Downloading {dataset_name} dataset to {dataset_path}...")
    
    # Download based on dataset type
    if dataset_name == 'tinyimagenet':
        _download_tinyimagenet(subdirectory, data_root)
    else:
        # Use torchvision for standard datasets
        _download_torchvision_dataset(display_name, dataset_class, subdirectory, has_train_test_split, data_root)
    
    print(f"✅ Dataset {dataset_name} downloaded successfully to {dataset_path}!")
    return dataset_path


def download_dataset(dataset_name, dataset_class, root_dir, has_train_test_split=True):
    """
    Download a dataset from torchvision.
    Used by main() function to download all datasets with verbose output.
    """
    full_root = os.path.join(BASE_ROOT, root_dir)
    os.makedirs(full_root, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"Directory: {full_root}")
    print(f"{'='*60}")
    
    try:
        success = _download_torchvision_dataset(dataset_name, dataset_class, root_dir, 
                                                has_train_test_split, BASE_ROOT, verbose=True)
        if success:
            print(f"{dataset_name} download complete!")
        return success
        
    except socket.timeout as e:
        print(f"✗ Timeout error downloading {dataset_name}: {e}")
        print(f"  This dataset may need to be downloaded manually or from a different source.")
        return False
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        print(f"  Error type: {type(e).__name__}")
        return False


def download_tinyimagenet(root_dir):
    """Download TinyImageNet dataset. Used by main() function."""
    return _download_tinyimagenet(root_dir, BASE_ROOT)


def main():
    """Main function to download all configured datasets."""
    print(f"\n{'='*60}")
    print("PyTorch Vision Datasets Downloader")
    print(f"{'='*60}")
    print(f"Base directory: {BASE_ROOT}")
    print(f"Datasets to download: {len(DATASET_CONFIG)}")
    print(f"{'='*60}\n")
    
    os.makedirs(BASE_ROOT, exist_ok=True)
    
    # Download each dataset
    results = {}
    for dataset_key, (display_name, dataset_class, root_dir, has_split) in DATASET_CONFIG.items():
        if dataset_class is None:
            # Handle custom downloads (TinyImageNet)
            if display_name == "TinyImageNet":
                success = download_tinyimagenet(root_dir)
            else:
                print(f"✗ Unknown custom dataset: {display_name}")
                success = False
        else:
            # Use torchvision download
            success = download_dataset(display_name, dataset_class, root_dir, has_split)
        results[display_name] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"\nSuccessful downloads ({len(successful)}):")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed downloads ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
        print("\nNote: Failed downloads may be due to network restrictions.")
        print("      Consider downloading manually or using a compute node with internet access.")
    
    print(f"\n{'='*60}")
    if failed:
        print("Some downloads failed. Check errors above.")
        return 1
    else:
        print("All datasets downloaded successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
