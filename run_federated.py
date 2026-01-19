#!/usr/bin/env python3
"""
Federated Learning Research Framework - Main Entry Point (Modular Version)
Supports: Multiple datasets, attacks, defenses, aggregation methods
Memory-optimized for CPU/GPU efficiency
"""

import argparse
import torch
import numpy as np
import random
from datetime import datetime
import wandb
from typing import Dict, Any
import yaml
from pathlib import Path

# Import from new modular core structure
from core import FLResearchFramework, MemoryMonitor
from data_loader import DataLoaderManager
from federated_trainer import FederatedTrainer


def setup_environment(config: Dict[str, Any]) -> None:
    """Setup environment with proper seeds and device configuration"""
    # Set random seeds
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

        # Set deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        device_id = config['gpu']['id']
        torch.cuda.set_device(device_id)
        print(f"🚀 Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        print("⚠️ CUDA not available, using CPU")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration directly from a YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")
    return config


def initialize_wandb(config: Dict[str, Any]) -> None:
    """Initialize Weights & Biases logging"""
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project'],
            name=f"{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            tags=config['experiment'].get('tags', [])
        )


def enhanced_memory_monitoring() -> None:
    """Enhanced memory monitoring using the new MemoryMonitor"""
    # Monitor memory at different stages
    MemoryMonitor.monitor_memory("Framework Initialization")
    
    # Get peak memory usage
    peak_cpu, peak_gpu = MemoryMonitor.get_round_peaks()
    print(f"📈 Peak CPU Memory: {peak_cpu:.2f}GB | Peak GPU Memory: {peak_gpu:.2f}GB")
    
    # Cleanup memory
    MemoryMonitor.cleanup_memory(aggressive=True)
    print("🧹 Memory cleanup completed")
    

def main():
    """Main entry point for the FL Research Framework (Modular Version)"""
    parser = argparse.ArgumentParser(description='FL Research Framework (Modular)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'analyze', 'demo'],
                       help='Mode: train, evaluate, analyze, or demo')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (overrides config)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Federated Learning Backdoor Attack Framework")
    print("=" * 60)
    
    # Load configuration
    config = load_yaml_config(args.config)
    
    # Override GPU if specified
    if args.gpu is not None:
        config['gpu']['id'] = args.gpu
    
    # Enable debug mode
    if args.debug:
        config['experiment']['debug'] = True
        print("🐛 Debug mode enabled")
    
    print(f"📋 Configuration: {config['experiment']['name']}")
    print(f"📊 Dataset: {config['dataset']['name']} ({config['dataset']['type']})")
    print(f"🏗️ Model: {config['model']['name']}")
    print(f"👥 Clients: {config['federated_learning']['num_clients']}")
    print(f"🔄 Rounds: {config['federated_learning']['num_rounds']}")
    
    # Setup environment
    setup_environment(config)
    
    # Initialize logging
    initialize_wandb(config)
    
    # Create framework components using new modular structure
    framework = FLResearchFramework(config)
    data_loader = DataLoaderManager(config)
    experiment_runner = FederatedTrainer(config, framework, data_loader)
    
    try:
        if args.mode == 'train':
            # Run training experiment
            print("\n🎯 Starting Training Experiment (Modular Version)")
            print("=" * 60)
            
            # Enhanced memory monitoring
            enhanced_memory_monitoring()
            results = experiment_runner.run_training()
            
            # Save results using experiment runner method
            results_path, json_path = experiment_runner.save_results(results, config['experiment']['name'])
            print(f"📁 Results saved to {results_path} and {json_path}")
        else:
            print(f"❌ Invalid mode: {args.mode}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"❌ Error during experiment: {str(e)}")
        if config['experiment'].get('debug', False):
            import traceback
            traceback.print_exc()
    finally:
        # Cleanup
        if config['logging']['use_wandb']:
            wandb.finish()
        
        # Final memory cleanup using new modular approach
        framework.cleanup_memory()
        MemoryMonitor.cleanup_memory(aggressive=True)
        print("\n🏁 Experiment completed!")


if __name__ == "__main__":
    main()
