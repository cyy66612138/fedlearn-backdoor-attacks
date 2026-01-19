# Federated Learning Backdoor Attack Framework

This repository provides a comprehensive framework for implementing and evaluating backdoor attacks in federated learning systems. The framework is designed to be modular and extensible, allowing researchers and developers to easily implement new attacks, defenses, and evaluation methodologies.

## 🎯 Overview

A comprehensive framework for implementing and evaluating backdoor attacks and defenses in federated learning. Designed to be modular and extensible, enabling researchers to easily implement new attacks, defenses, and evaluation methodologies. The framework provides multiple attack implementations, defense mechanisms, and tools for comprehensive evaluation across various datasets.

## 🚀 Key Features

- **Modular Architecture**: Easy-to-extend base classes for attacks and defenses
- **Multiple Attack Types**: Pattern-based, model poisoning, and label manipulation attacks
- **Comprehensive Defenses**: Robust aggregation methods and defense mechanisms
- **Flexible Configuration**: YAML-based configuration for experiment management
- **Reproducibility**: Fixed seeds and comprehensive logging
- **Multi-Dataset Support**: CIFAR-10/100, MNIST, Fashion-MNIST, SVHN, GTSRB, TinyImageNet

## 🛠️ Installation

**Note**: This project supports Python 3.9, 3.10, 3.11, and 3.12. Tested with Python 3.11.13.

### Setup with uv (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd <project-directory>

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.11  # Or 3.9, 3.10, 3.12 (tested with 3.11.13)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (using pyproject.toml)
uv pip install -e .

# Generate lock file with exact versions for reproducibility
uv lock

# For reproducible installation (if uv.lock exists):
# uv sync  # Installs exact versions from uv.lock (works with any supported Python version)
```

**Note on `uv.lock`**: The lock file supports Python 3.9-3.12. It includes resolution markers that select appropriate package versions for each Python version. For maximum reproducibility, use the same Python version (3.11.13) as tested.

### Alternative: pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # Fallback for pip users
```

**Note**: With `uv`, `pyproject.toml` is the primary source. `requirements.txt` is kept for compatibility with pip users.

## 🎯 Quick Start

### Example: Running a Backdoor Attack Experiment

**Step 1: Generate experiment configuration**

```bash
python gen_exps_config.py \
  --attack badnets \
  --base configs/base.yaml \
  --output configs/generated \
  --dataset cifar10 \
  --aggregation FedAvg
```

This generates a config file in `configs/generated/` with parameters for the attack.

**Step 2: Run federated learning experiment**

```bash
python run_federated.py \
  --config ./configs/badnets_cifar10.yaml \
  --gpu 0
```

**Step 3: View results**

Results are saved in:
- `results/`: Training metrics and history
- `checkpoints/`: Model checkpoints
- `logs/`: Training logs
- `visualizations/`: Attack visualizations and trigger examples

### Running with Different Aggregation Methods

Evaluate attacks against various robust aggregation methods:

```bash
# Generate configs for multiple aggregation methods
python gen_exps_config.py \
  --attack badnets \
  --base configs/base.yaml \
  --output configs/generated \
  --dataset cifar10 \
  --aggregation FedAvg SCAFFOLD FedOpt Median Krum TrimmedMean

# Run each experiment
python run_federated.py --config configs/generated/... --gpu 0
```

### Comparing Multiple Attacks

Compare different attack strategies:

```bash
# Generate configs for multiple attacks
python gen_exps_config.py \
  --attack badnets blended dba sinusoidal \
  --base configs/base.yaml \
  --output configs/generated \
  --dataset cifar10 \
  --aggregation FedAvg
```

## 📊 Supported Components

### Datasets
CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN, GTSRB, TinyImageNet

### Available Attacks

**Pattern-Based Attacks**:
- **BadNets**: Static trigger pattern attack
- **Blended**: Blended trigger pattern attack
- **Sinusoidal**: Sinusoidal pattern attack
- **DBA**: Distributed Backdoor Attack with multiple local triggers

**Model Poisoning Attacks**:
- **ModelReplacement**: Model replacement/scaling attack
- **Neurotoxin**: Gradient masking attack
- **EdgeCaseBackdoor**: Edge-case sample attack
- **ThreeDFed**: Covert backdoor with norm clipping

**Other Attacks**:
- **LabelFlipping**: Label manipulation attack

### Aggregation Methods & Defenses

**Traditional FL Aggregation**: 
FedAvg, FedSGD, FedProx, SCAFFOLD, FedOpt

**Robust Aggregation Methods**: 
Median, CoordinateWiseMedian, TrimmedMean, Krum, MultiKrum, Bulyan, RFA

**Defense Methods**: 
FLAME, DeepSight, FLDetector, FLTrust, FoolsGold, RLR, MultiMetric, DnC, FLARE, LASA, Bucketing, AUROR, SignGuard, NormClipping, WeakDP, CRFL, CenteredClipping

## 🔧 Extending the Framework

### Adding a New Attack

To implement a new backdoor attack:

1. **Create your attack class** in `core/attacks.py`:

```python
from .attacks import BaseAttack

class MyCustomAttack(BaseAttack):
    """Your custom attack implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize your attack-specific parameters
        
    def get_data_type(self) -> str:
        return "image"  # or "time_series"
    
    def _apply_static_trigger(self, poisoned_data, poisoned_labels, poison_indices):
        # Implement your trigger application logic
        pass
    
    # Or override _generate_attack_batch for generative attacks
```

2. **Register your attack** in the `create_attack()` factory function:

```python
def create_attack(attack_config: Dict[str, Any]) -> BaseAttack:
    attack_name = attack_config['name']
    
    if attack_name == 'MyCustomAttack':
        return MyCustomAttack(attack_config)
    # ... other attacks
```

3. **Add configuration support** in `gen_exps_config.py` if needed

### Adding a New Defense

To implement a new defense mechanism:

1. **Create your aggregation class** in `core/aggregations.py`:

```python
from .aggregations import BaseAggregation

class MyCustomDefense(BaseAggregation):
    """Your custom defense implementation"""
    
    def aggregate(self, client_updates, client_weights):
        # Implement your aggregation logic
        pass
```

2. **Register your defense** in the aggregation factory

3. **Update configuration** to include your defense

### Adding a New Dataset

1. **Add dataset loading logic** in `data_loader.py`
2. **Update normalization parameters** in configuration files
3. **Add dataset-specific model architectures** if needed in `core/custom_models/`

## ⚙️ Configuration

Configs are YAML files. Generate experiment configs using `gen_exps_config.py`:

```bash
python gen_exps_config.py \
  --attack <attack_name> \
  --base configs/base.yaml \
  --output configs/generated \
  --dataset <dataset> \
  --aggregation <agg_method>
```

**Supported attacks**: `badnets`, `blended`, `sinusoidal`, `dba`, `labelflipping`, `modelreplacement`, `neurotoxin`, `edgecasebackdoor`, `threedfed`

**Supported datasets**: `cifar10`, `cifar100`, `mnist`, `fashionmnist`, `svhn`, `gtsrb`, `tinyimagenet`

**Supported aggregations**: See list above (can specify multiple)

## 📈 Output Structure

```
<project-directory>/
├── results/          # Training metrics and history
├── checkpoints/     # Model checkpoints (including attack models)
├── logs/            # Training logs
├── visualizations/  # Attack visualizations
├── configs/         # Configuration files
│   ├── base.yaml    # Base configuration
│   └── generated/   # Generated experiment configs
└── core/            # Core framework code
    ├── attacks.py   # Attack implementations
    ├── aggregations.py  # Defense/aggregation implementations
    ├── client.py    # Client-side FL logic
    ├── server.py    # Server-side FL logic
    └── ...
```

## 🔬 Reproducibility

All experiments use fixed random seeds for reproducibility. The framework includes:
- Deterministic training with seed configuration
- Checkpoint saving/loading for experiment resumption
- Comprehensive logging of all hyperparameters and results
- Version tracking for dependencies

## 📚 Code Structure

- `core/attacks.py`: Base attack class and all attack implementations
- `core/aggregations.py`: Aggregation methods and defense mechanisms
- `core/client.py`: Client-side federated learning logic
- `core/server.py`: Server-side federated learning logic
- `core/framework.py`: Core federated learning framework
- `core/models.py`: Model definitions
- `data_loader.py`: Dataset loading utilities
- `federated_trainer.py`: Main training loop
- `run_federated.py`: Entry point for running experiments
- `gen_exps_config.py`: Configuration file generator

## 🤝 Contributing

This framework is designed to be extended. When adding new features:

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings
3. Update this README with new features
4. Ensure reproducibility with proper seed handling
5. Add appropriate error handling

## 📄 License

MIT License

## 🙏 Acknowledgments

This framework implements various attacks and defenses from the federated learning security literature. For a comprehensive survey on backdoor attacks and defenses in federated learning, please refer to:

Nguyen, T. D., Nguyen, T., Le Nguyen, P., Pham, H. H., Doan, K. D., & Wong, K. S. (2024). Backdoor attacks and defenses in federated learning: Survey, challenges and future research directions. *Engineering Applications of Artificial Intelligence*, 127, 107166. https://doi.org/10.1016/j.engappai.2023.107166

Please cite the original papers when using specific attacks or defenses in your research.

## 📚 Additional Resources

**Backdoor Learning Papers (Up-to-date)**:
- GitHub Repository: [backdoor-ai-resources](https://github.com/mtuann/backdoor-ai-resources) - Comprehensive collection of backdoor learning papers with code
- Interactive Search & Browse: [Research Papers Portal](https://mtuann.github.io/papers/) - Filter, search, and explore all papers with an intuitive interface
