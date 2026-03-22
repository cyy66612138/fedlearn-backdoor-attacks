#!/usr/bin/env python3
"""
Unified config generators for multiple backdoor attacks.

Provides one function per attack and a single CLI entrypoint.
"""

import argparse
import os
from typing import Dict, List
import yaml
from datetime import datetime
import random
import numpy as np


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_optimizer(dataset_name, opt_type='SGD'):
    dataset_name = dataset_name.lower()
    # SGD configurations
    sgd_configs = {
        'mnist': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0},
        'fashionmnist': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0},
        'cifar10': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
        'cifar100': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
        'tinyimagenet': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
        'gtsrb': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
        'svhn': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4},
    }

    # Adam configurations
    adam_configs = {
        'mnist': {'lr': 0.001, 'weight_decay': 0},
        'fashionmnist': {'lr': 0.001, 'weight_decay': 0},
        'cifar10': {'lr': 0.001, 'weight_decay': 1e-4},
        'cifar100': {'lr': 0.001, 'weight_decay': 1e-4},
        'tinyimagenet': {'lr': 0.001, 'weight_decay': 1e-4},
        'gtsrb': {'lr': 0.001, 'weight_decay': 1e-4},
        'svhn': {'lr': 0.001, 'weight_decay': 1e-4},
    }

    return sgd_configs.get(dataset_name) if opt_type.upper() == 'SGD' else adam_configs.get(dataset_name)


# Dataset configurations (updated with computed normalization stats)
DATASET_CONFIGS = {
    'mnist': {
        'normalization': ((0.1307,), (0.3081,)),
        'num_classes': 10,
        'model_name': 'simplecnn',
        'optimizer': 'SGD',
        'data_shape': (1, 28, 28),
    },
    'fashionmnist': {
        'normalization': ((0.2860,), (0.3530,)),
        'num_classes': 10,
        'model_name': 'simplecnn',
        'optimizer': 'SGD',
        'data_shape': (1, 28, 28),
    },
    'cifar10': {
        'normalization': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        'num_classes': 10,
        'model_name': 'resnet18',
        'optimizer': 'Adam',
        'data_shape': (3, 32, 32),
    },
    'cifar100': {
        'normalization': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        'num_classes': 100,
        'model_name': 'resnet18',
        'optimizer': 'Adam',
        'data_shape': (3, 32, 32),
    },
    'gtsrb': {
        'normalization': ((0.3417, 0.3126, 0.3216), (0.2737, 0.2607, 0.2662)),
        'num_classes': 43,
        'model_name': 'resnet18',
        'optimizer': 'Adam',
        'data_shape': (3, 32, 32),
    },
    'tinyimagenet': {
        'normalization': ((0.4802, 0.4481, 0.3975), (0.2764, 0.2689, 0.2816)),
        'num_classes': 200,
        'model_name': 'resnet18',
        'optimizer': 'Adam',
        'data_shape': (3, 64, 64),
    },
    'femnist': {
        'normalization': ((0.1722,), (0.3309,)),
        'num_classes': 62,
        'model_name': 'simplecnn',
        'optimizer': 'SGD',
        'data_shape': (1, 28, 28),
    },
    'svhn': {
        'normalization': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        'num_classes': 10,
        'model_name': 'resnet18',
        'optimizer': 'Adam',
        'data_shape': (3, 32, 32),
    },
}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clone_config(config: dict) -> dict:
    # Shallow copy is sufficient given the simple edits we perform
    return {k: (v.copy() if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in config.items()}


def set_experiment_name(config: dict, suffix: str, dataset: str = 'cifar10') -> None:
    if 'experiment' in config and isinstance(config['experiment'], dict):
        # model_name = DATASET_CONFIGS.get(dataset, {}).get('model_name', 'resnet18')  # default to resnet18 if not found
        config['experiment']['name'] = f"{suffix}"

    # set logging wandb
    if 'logging' in config and isinstance(config['logging'], dict):
        # suffix will have name of the attack, dataset, and model (base_fashionmnist_simplecnn)
        # config, loss, accuracy, etc. will be saved in the results directory
        type_experiment = suffix.split('_')[0]
        config['logging']['save_results'] = True
        config['logging']['save_results_dir'] = f"./results/{type_experiment}"
        config['logging']['save_visualizations'] = True
        config['logging']['visualize_dir'] = f"./visualizations/{type_experiment}"

        # checkpoints will be saved in the checkpoints directory
        config['logging']['save_checkpoints'] = True
        config['logging']['checkpoint_frequency'] = 100
        config['logging']['checkpoint_dir'] = f"./checkpoints/{type_experiment}"

        # wandb will be used to log the results
        config['logging']['use_wandb'] = False
        config['logging']['project'] = "fedlearn-backdoor"


def set_dataset_normalization_and_optimizer(config: dict, dataset: str, opt_type: str = "SGD",
                                            alpha_non_iid: float = 0.5) -> None:
    """Set dataset-specific normalization values, num_classes, and model in the config"""
    if dataset in DATASET_CONFIGS:
        dataset_config = DATASET_CONFIGS[dataset]
        # Update data_loader normalization if it exists
        mean, std = dataset_config['normalization']
        config['dataset']['name'] = dataset
        config['dataset']['mean'] = list(mean)
        config['dataset']['std'] = list(std)
        config['dataset']['alpha'] = alpha_non_iid
        config['dataset']['num_classes'] = dataset_config['num_classes']
        config['dataset']['data_shape'] = list(dataset_config['data_shape'])

        # Update model configuration if it exists
        config['model']['name'] = dataset_config['model_name']

        # update optimizer
        opt_config = get_optimizer(dataset, opt_type)
        config['federated_learning']['optimizer'] = opt_type
        config['federated_learning']['learning_rate'] = opt_config['lr']
        config['federated_learning']['weight_decay'] = opt_config['weight_decay']
        if 'momentum' in opt_config:
            config['federated_learning']['momentum'] = opt_config['momentum']


def write_config(config: dict, output_dir: str, filename: str, attack_name: str = None) -> None:
    attack_dir = os.path.join(output_dir, attack_name)
    os.makedirs(attack_dir, exist_ok=True)
    output_path = os.path.join(attack_dir, filename)
    # save_config(config, output_path)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Writing config to: {output_path}")
    return output_path


def set_fl_aggregation(config: dict, aggregation_name: str, dataset: str = 'cifar10') -> None:
    if 'server_aggregation' in config:
        config['server_aggregation'] = []
        if aggregation_name in ['FedAvg', 'Median']:
            config['server_aggregation'].append({'name': aggregation_name, 'params': {}})
        elif aggregation_name == 'FedSGD' or aggregation_name == 'fedsgd':
            # FedSGD: Federated Stochastic Gradient Descent
            # Uses same aggregation as FedAvg, but clients perform single local epoch
            # Server-side aggregation is identical to FedAvg
            config['server_aggregation'].append({'name': aggregation_name, 'params': {}})
        elif aggregation_name == 'FedProx' or aggregation_name == 'fedprox':
            # FedProx: Uses same aggregation as FedAvg, but client-side adds proximal term
            # Server-side aggregation is identical to FedAvg
            config['server_aggregation'].append({'name': aggregation_name, 'params': {}})
        elif aggregation_name == 'SCAFFOLD' or aggregation_name == 'scaffold':
            # SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
            # Default: global_lr=1.0 (server learning rate for aggregating updates)
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'global_lr': 1.0
            }})
        elif aggregation_name == 'FedOpt' or aggregation_name == 'fedopt':
            # FedOpt: Adaptive Federated Optimization
            # Default: type='adam', beta1=0.9, beta2=0.999, server_lr=0.1, tau=1e-3
            # Note: type can be 'adam', 'adagrad', or 'yogi'
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'type': 'adam',
                'beta1': 0.9,
                'beta2': 0.999,
                'server_lr': 0.1,
                'tau': 1e-3
            }})
        elif aggregation_name in ['Krum', 'MultiKrum']:
            num_adversarial_clients = len(config['adversarial_clients'])
            avg_percentage = 0.2 if aggregation_name == 'MultiKrum' else 1
            config['server_aggregation'].append(
                {'name': aggregation_name, 'params': {'f': num_adversarial_clients, 'avg_percentage': avg_percentage}})
        elif aggregation_name == 'NormClipping':
            config['server_aggregation'].append({'name': aggregation_name,
                                                 'params': {'norm_threshold': 3, 'weakDP': False, 'noise_mean': 0,
                                                            'noise_std': 0.002}})
        elif aggregation_name == 'WeakDP':
            config['server_aggregation'].append({'name': aggregation_name,
                                                 'params': {'norm_threshold': 3, 'weakDP': True, 'noise_mean': 0,
                                                            'noise_std': 0.001}})
        elif aggregation_name == 'CRFL':
            config['server_aggregation'].append(
                {'name': aggregation_name, 'params': {'norm_threshold': 3, 'noise_mean': 0, 'noise_std': 0.001}})
        elif aggregation_name == 'Bulyan':
            total_clients = int(
                config['federated_learning']['num_clients'] * config['federated_learning']['participation_rate'])
            num_adversarial_clients = len(config['adversarial_clients'])
            beta = max(1, int(total_clients - 2 * num_adversarial_clients))
            bul_params = {'f': num_adversarial_clients, 'beta': beta}
            print(f"🔍 Bulyan with config: {bul_params}")
            config['server_aggregation'].append(
                {'name': aggregation_name, 'params': {'f': num_adversarial_clients, 'beta': beta}})
        elif aggregation_name == 'CoordinateWiseMedian':
            config['server_aggregation'].append({'name': aggregation_name, 'params': {}})
        elif aggregation_name == 'TrimmedMean':
            config['server_aggregation'].append({'name': aggregation_name, 'params': {'proportion': 0.1}})
        elif aggregation_name == 'FLTrust':
            # Server trust anchor based weighting; params left minimal here
            config['server_aggregation'].append({'name': aggregation_name, 'params': {'epsilon': 1e-9}})
        elif aggregation_name == 'FLDetector':
            # History-based detection and filtering
            config['server_aggregation'].append(
                {'name': aggregation_name, 'params': {'window_size': 10, 'start_epoch': 50}})
        elif aggregation_name == 'Flame':
            # FLAME: clustering + clipping + DP-like noise
            config['server_aggregation'].append({'name': aggregation_name, 'params': {'gamma': 1.2e-5}})
        elif aggregation_name == 'DeepSight':
            # DeepSight: NEUP/TEs + DDifs + cosine on biases, ensemble clustering
            data_shape = DATASET_CONFIGS[dataset]['data_shape']
            # might add dataset: CINIC10, CHMNIST if needed
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'num_seeds': 3,
                'threshold_factor': 0.01,
                'num_samples': 20000,
                'tau': 0.33,
                'epsilon': 1.0e-6,
                # Dataset-related defaults; override in configs if needed
                'num_channels': data_shape[0],
                'num_dims': data_shape[1],
                'batch_size': 128,  # Compute DDifs with random images (torch.rand(20000, 3, 32, 32))
                'num_workers': 0,
                'device': 'cuda',
            }})
        elif aggregation_name == 'RFA':
            config['server_aggregation'].append(
                {'name': aggregation_name, 'params': {'num_iters': 3, 'epsilon': 1.0e-6}})
        elif aggregation_name == 'CenteredClipping':
            config['server_aggregation'].append({'name': aggregation_name, 'params': {'norm_threshold': -1.0}})
        elif aggregation_name == 'SimpleClustering':
            config['server_aggregation'].append({'name': aggregation_name, 'params': {'num_clusters': 10}})
        elif aggregation_name == 'FoolsGold':
            config['server_aggregation'].append(
                {'name': aggregation_name, 'params': {'epsilon': 1.0e-6, 'topk_ratio': 0.1}})
        elif aggregation_name == 'RLR':
            # RLR: Robust Learning Rate - sign-based aggregation with adaptive learning rates
            # Default: server_lr=1.0, robustLR_threshold=None (uses num_clients/2)
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'server_lr': 1.0,
                'robustLR_threshold': None  # None means use num_clients/2
            }})
        elif aggregation_name == 'MultiMetric' or aggregation_name == 'mm':
            # MultiMetric: Multi-dimensional anomaly detection using Mahalanobis distance
            # Default: proportion=0.3 (30% of clients selected), epsilon=1e-6
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'proportion': 0.3,
                'epsilon': 1e-6
            }})
        elif aggregation_name == 'DnC' or aggregation_name == 'dnc':
            # DnC: Divide and Conquer - Robust aggregation using iterative SVD-based filtering
            # Default: num_byzantine from adversarial_clients, sub_dim=10000, num_iters=5, filter_frac=1.0
            num_adversarial_clients = len(config['adversarial_clients'])
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'num_byzantine': num_adversarial_clients if num_adversarial_clients > 0 else None,
                'sub_dim': 10000,
                'num_iters': 5,
                'filter_frac': 1.0,
                'byzantine_ratio': 0.1
            }})
        elif aggregation_name == 'FLARE':
            # FLARE: Feature-based defense using MMD and voting
            # Default: k_ratio=0.5, sigma=1.0, batch_size=64
            # Note: central_dataset must be provided separately via kwargs or config
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'k_ratio': 0.5,
                'sigma': 1.0,
                'epsilon': 1e-6,
                'batch_size': 64
            }})
        elif aggregation_name == 'LASA':
            # LASA: Layer-wise Adaptive Secure Aggregation
            # Dataset-specific: norm_bound=1.0 for CIFAR10/100, 2.0 otherwise
            norm_bound = 1.0 if dataset.lower() in ['cifar10', 'cifar100'] else 2.0
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'norm_bound': norm_bound,
                'sign_bound': 1.0,
                'sparsity': 0.3
            }})
        elif aggregation_name == 'Bucketing':
            # Bucketing: Byzantine-Robust Learning via Bucketing
            # Default: bucket_size=5, selected_aggregator='Median'
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'bucket_size': 5,
                'selected_aggregator': 'Median',
                'selected_aggregator_params': {}
            }})
        elif aggregation_name == 'AUROR':
            # AUROR: Defending against poisoning attacks
            # Dataset-specific thresholds: 1e-4 for MNIST, 7e-4 for CIFAR10, 0.002 default
            if dataset.lower() == 'mnist':
                indicative_threshold = 1e-4
            elif dataset.lower() == 'cifar10':
                indicative_threshold = 7e-4
            else:
                indicative_threshold = 0.002
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'indicative_threshold': indicative_threshold,
                'indicative_find_epoch': 10
            }})
        elif aggregation_name == 'SignGuard':
            # SignGuard: Byzantine-robust FL through collaborative malicious gradient filtering
            # Default: lower_bound=0.1, upper_bound=3.0, selection_fraction=0.1, clustering='DBSCAN'
            config['server_aggregation'].append({'name': aggregation_name, 'params': {
                'lower_bound': 0.1,
                'upper_bound': 3.0,
                'selection_fraction': 0.1,
                'clustering': 'DBSCAN',
                'random_seed': 0
            }})
        elif aggregation_name == 'Mean':
            # Mean: Simple average aggregation (equal weights for all clients)
            config['server_aggregation'].append({'name': aggregation_name, 'params': {}})
        else:
            raise ValueError(f"Aggregation name {aggregation_name} not supported")


# def set_pattern_attack_configs(config: dict, dataset: str, attack_type: str, target_label: int, id_adversarial_clients: List[int]) -> None:
#     # for base then we don't need to set any attack configs
#     config['client_attacks'] = []
#
#     if attack_type.lower() == "badnets":
#         badnets_config = {
#             '28': {
#                 'trigger_height': 4,
#                 'trigger_width': 4,
#             },
#             '32': {
#                 'trigger_height': 5,
#                 'trigger_width': 5,
#             },
#             '64': {
#                 'trigger_height': 9,
#                 'trigger_width': 9,
#             }
#         }
#         input_dim = DATASET_CONFIGS[dataset]['data_shape'][1] # width/height of the dataset
#         trigger_config = badnets_config[str(input_dim)]
#         config['client_attacks'].append({
#             'name': 'BadNetsAttack',
#             'trigger_height': trigger_config['trigger_height'],
#             'trigger_width': trigger_config['trigger_width'],
#             'trigger_pattern': 'square',
#             'target_class': target_label,
#             'poison_ratio': 0.5,
#             'apply_to_client_ids': id_adversarial_clients,
#         })
#     elif attack_type.lower() == "sinusoidal":
#         config['client_attacks'].append({
#             'name': 'SinusoidalAttack',
#             'target_class': target_label,
#             'sine_amplitude': 0.2,
#             'sine_frequency': 4.0,
#             'sine_phase': 0.0,
#             'sine_orientation': 'horizontal',  # or 'vertical' # or 'single'
#             'channel_mode': 'all',             # RGB or grayscale
#             'poison_ratio': 0.5,
#             'apply_to_client_ids': id_adversarial_clients,
#         })
#     elif attack_type.lower() == "blended":
#         config['client_attacks'].append({
#             'name': 'BlendedAttack',
#             'target_class': target_label,
#             'blend_alpha': 0.1,
#             'poison_ratio': 0.5,
#             'apply_to_client_ids': id_adversarial_clients,
#         })
#     elif attack_type.lower() == "dba":
#         # DBA trigger parameters based on dataset input dimensions (horizontal triggers)
#         # https://github.com/AI-secure/DBA/blob/master/utils/cifar_params.yaml
#         DBA_TRIGGER_CONFIGS = {
#             '28': {  # MNIST, FashionMNIST, FEMNIST
#                 'trigger_height': 1,   # Horizontal: height < width
#                 'trigger_width': 4,
#                 'trigger_gap': 2,
#             },
#             '32': {  # CIFAR10, CIFAR100, GTSRB
#                 'trigger_height': 1,   # Horizontal: height < width
#                 'trigger_width': 6,
#                 'trigger_gap': 3,
#             },
#             '64': {  # TinyImageNet
#                 'trigger_height': 2,   # Horizontal: height < width
#                 'trigger_width': 10,
#                 'trigger_gap': 2,
#             }
#         }
#         input_dim = DATASET_CONFIGS[dataset]['data_shape'][1]
#
#         # Get trigger parameters for this dataset
#         trigger_config = DBA_TRIGGER_CONFIGS[str(input_dim)]
#         config['client_attacks'].append({
#             'name': 'DBAAttack',
#             'target_class': target_label,
#             'trigger_num': len(id_adversarial_clients),
#             'trigger_height': trigger_config['trigger_height'],
#             'trigger_width': trigger_config['trigger_width'],
#             'trigger_gap': trigger_config['trigger_gap'],
#             'trigger_shift': 0,
#             'poison_ratio': 0.5,
#             'apply_to_client_ids': id_adversarial_clients,
#             'use_global_backdoor': False,
#         })
#     else:
#         raise ValueError(f"Attack type {attack_type} not supported")

def set_pattern_attack_configs(config: dict, dataset: str, attack_type: str, target_label: int,
                               id_adversarial_clients: List[int]) -> None:
    # for base then we don't need to set any attack configs
    config['client_attacks'] = []

    input_dim = DATASET_CONFIGS[dataset]['data_shape'][1]  # width/height of the dataset

    # Auto-set trigger dimensions based on input size
    if input_dim == 28:
        default_size = 4
    elif input_dim == 32:
        default_size = 5
    elif input_dim == 64:
        default_size = 9
    else:
        default_size = 5

    if attack_type.lower() == "badnets":
        config['client_attacks'].append({
            'name': 'BadNetsAttack',
            'trigger_height': default_size,
            'trigger_width': default_size,
            'trigger_pattern': 'square',
            'target_class': target_label,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "neurotoxin":
        config['client_attacks'].append({
            'name': 'NeurotoxinAttack',
            'target_class': target_label,
            'topk_ratio': 0.1,  # 寻找 10% 的休眠神经元
            'lambda_val': 2.0,  # [新增] 掩码后的能量放大倍数 (代码会自动进行安全裁剪，不用担心超标)
            'trigger_position': 'bottom-right',
            'trigger_height': default_size,
            'trigger_width': default_size,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    # elif attack_type.lower() == "feddare":
    #     config['client_attacks'].append({
    #         'name': 'FedDAREAttack',
    #         'target_class': target_label,
    #         'drop_rate': 0.99,
    #         'trigger_height': default_size,
    #         'trigger_width': default_size,
    #         'poison_ratio': 0.5,
    #         'apply_to_client_ids': id_adversarial_clients,
    #     })
    # elif attack_type.lower() == "feddare":
    #     config['client_attacks'].append({
    #         'name': 'FedDAREAttack',
    #         'target_class': target_label,
    #         'drop_rate': 0.5,
    #         'cos_tau': 0.7,  # O-FedDARE 正交双约束：死死锁定余弦相似度
    #         'trigger_height': default_size,
    #         'trigger_width': default_size,
    #         'poison_ratio': 0.5,
    #         'apply_to_client_ids': id_adversarial_clients,
    #     })
    # elif attack_type.lower() == "feddare":
    #     config['client_attacks'].append({
    #         'name': 'FedDAREAttack',
    #         'target_class': target_label,
    #         'drop_rate': 0.99,  # 必须是 0.99！精准搭乘那 1% 的最强良性顺风车
    #         'cos_tau': 0.99,  # 必须是 0.99！保持绝对的几何隐蔽，免疫 Flame 和 FLTrust
    #         'trigger_height': default_size,
    #         'trigger_width': default_size,
    #         'poison_ratio': 0.5,
    #         'apply_to_client_ids': id_adversarial_clients,
    #     })
    # elif attack_type.lower() == "feddare":
    #     config['client_attacks'].append({
    #         'name': 'FedDAREAttack',
    #         'target_class': target_label,
    #         'drop_rate': 0.99,  # 依然保持 1% 的微小截面积，降低被防守方发现的物理概率
    #         'gamma': 50.0,  # 【核心火力】起步直接给 50 倍！(甚至可以给到 100)
    #         'trigger_height': default_size,
    #         'trigger_width': default_size,
    #         'poison_ratio': 0.5,
    #         'apply_to_client_ids': id_adversarial_clients,
    #     })
    elif attack_type.lower() == "feddare":
        config['client_attacks'].append({
            'name': 'FedDAREAttack',
            'target_class': target_label,

            # 【方案一核心】
            'drop_rate': 0.99,
            # 【方案一核心：人为放大】
            'gamma': 10.0,

            'trigger_height': default_size,
            'trigger_width': default_size,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "modelreplacement":
        config['client_attacks'].append({
            'name': 'ModelReplacementAttack',
            'target_class': target_label,
            'scaling_factor': 10,
            'alpha': 0.5,
            'trigger_position': 'bottom-right',
            'trigger_height': default_size,
            'trigger_width': default_size,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "threedfed":
        config['client_attacks'].append({
            'name': 'ThreeDFedAttack',
            'target_class': target_label,
            'scaling_factor': 1.0,
            'use_norm_clipping': True,
            'trigger_position': 'bottom-right',
            'trigger_height': default_size,
            'trigger_width': default_size,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "edgecase":
        config['client_attacks'].append({
            'name': 'EdgeCaseBackdoorAttack',
            'target_class': 9,
            'dataset_name': dataset,
            'data_root': './data',
            'epsilon': 0.25 if dataset.lower() in ['mnist', 'fashionmnist'] else 0.083,
            'projection_type': 'l_2',
            'PGD_attack': True,
            'scaling_attack': True,
            'scaling_factor': 50,
            'l2_proj_frequency': 1,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "labelflipping":
        config['client_attacks'].append({
            'name': 'LabelFlippingAttack',
            'attack_model': 'targeted',
            'source_label': (target_label + 1) % DATASET_CONFIGS[dataset]['num_classes'],
            'target_label': target_label,
            'num_classes': DATASET_CONFIGS[dataset]['num_classes'],
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "sinusoidal":
        config['client_attacks'].append({
            'name': 'SinusoidalAttack',
            'target_class': target_label,
            'sine_amplitude': 0.2,
            'sine_frequency': 4.0,
            'sine_phase': 0.0,
            'sine_orientation': 'horizontal',
            'channel_mode': 'all',
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "blended":
        config['client_attacks'].append({
            'name': 'BlendedAttack',
            'target_class': target_label,
            'blend_alpha': 0.1,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "dba":
        DBA_TRIGGER_CONFIGS = {
            '28': {'trigger_height': 1, 'trigger_width': 4, 'trigger_gap': 2},
            '32': {'trigger_height': 1, 'trigger_width': 6, 'trigger_gap': 3},
            '64': {'trigger_height': 2, 'trigger_width': 10, 'trigger_gap': 2}
        }
        trigger_config = DBA_TRIGGER_CONFIGS[str(input_dim)]
        config['client_attacks'].append({
            'name': 'DBAAttack',
            'target_class': target_label,
            'trigger_num': len(id_adversarial_clients),
            'trigger_height': trigger_config['trigger_height'],
            'trigger_width': trigger_config['trigger_width'],
            'trigger_gap': trigger_config['trigger_gap'],
            'trigger_shift': 0,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
            'use_global_backdoor': False,
        })
    # elif attack_type.lower() == "layerwisepoisoning":
    #     config['client_attacks'].append({
    #         'name': 'LayerwisePoisoningAttack',
    #         'bc_layers': ['layer4', 'linear', 'fc'],
    #         'lambda_val': 2.0,
    #         'trigger_height': default_size,
    #         'trigger_width': default_size,
    #         'poison_ratio': 0.5,
    #         'target_class': target_label,
    #         'apply_to_client_ids': id_adversarial_clients,
    #     })
    elif attack_type.lower() == "layerwisepoisoning":
        config['client_attacks'].append({
            'name': 'LayerwisePoisoningAttack',
            'target_class': target_label,
            'bc_layer_ratio': 0.05,  # [关键] 只取 BSR 最高的 Top 5% 参数层作为关键层（对 ResNet18 大约是 3 层）
            'lambda_val': 2.0,  # 针对找出来的 BC 层，把后门更新量放大 2 倍
            'lsa_bsr_threshold': 0.5,  # LSA 备用安全阈值
            'trigger_height': default_size,
            'trigger_width': default_size,
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })

    elif attack_type.lower() == "minmax":
        config['client_attacks'].append({
            'name': 'MinMaxAttack',
            'dev_type': 'std',  # 采用论文中效果最好的反向标准差方向
            'poison_ratio': 1.0,  # Min-Max 是模型级投毒，不需要数据层面的 poison_ratio
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "trim":
        config['client_attacks'].append({
            'name': 'TrimAttack',
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "krum":
        config['client_attacks'].append({
            'name': 'KrumAttack',
            'poison_ratio': 0.5,
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "cerp":
        config['client_attacks'].append({
            'name': 'CerPAttack',
            'target_class': target_label,
            'trigger_height': 6,  # CerP 需要稍微大一点的触发器区域以便切分
            'trigger_width': 6,
            'poison_ratio': 0.5,
            'epsilon': 2.0,  # 控制隐蔽性的 L2 范数阈值 (可视防御算法的严格程度调节)
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "a3fl":
        config['client_attacks'].append({
            'name': 'A3FLAttack',
            'target_class': target_label,
            'trigger_height': 5,
            'trigger_width': 5,
            'adv_epochs': 3,  # 对抗优化触发器的迭代轮数
            'poison_ratio': 0.5,
            'scaling_factor': 1,  # A3FL主要靠强大的触发器，不强制需要大缩放倍数
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "fcba":
        config['client_attacks'].append({
            'name': 'FCBAAttack',
            'target_class': target_label,
            'trigger_height': 3,  # 因为是四个角组合，单个触发器块可以稍微小一点
            'trigger_width': 3,
            'poison_ratio': 0.5,
            'scaling_factor': 5,  # 适度放大，巩固组合特征在全局模型中的权重
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "iba":
        config['client_attacks'].append({
            'name': 'IBAAttack',
            'target_class': target_label,
            'adv_epochs': 2,  # 生成器在每轮开始前的预训练轮数
            'epsilon': 0.15,  # 控制对抗噪声触发器的隐蔽性上限
            'pgd_bound': 2.0,  # 控制局部模型更新不被防御算法(如Krum)当成异常值踢出的安全半径
            'poison_ratio': 0.5,
            'scaling_factor': 5,  # 在 PGD 约束下安全放大的倍数，用于延长后门寿命
            'apply_to_client_ids': id_adversarial_clients,
        })
    elif attack_type.lower() == "darkfed":
        config['client_attacks'].append({
            'name': 'DarkFedAttack',
            'target_class': target_label,
            'poison_ratio': 0.5,
            'trigger_width': default_size,
            'safe_norm_bound': 2.0,
            'apply_to_client_ids': id_adversarial_clients,
        })
    else:
        raise ValueError(f"Attack type {attack_type} not supported")


def set_fl_attack_clients(config: dict, attack_type: str, num_clients: int, start_attack_round: int,
                          stop_attack_round: int, attack_frequency: int, num_rounds: int):
    config['federated_learning']['num_rounds'] = num_rounds
    # participation_rate = config['federated_learning']['participation_rate']
    # num_clients_per_round = int(num_clients * participation_rate)
    # number_adversarial_clients = random.randint(1, num_clients_per_round)

    config['experiment']['description'] = f"Federated learning experiments evaluating the {attack_type.upper()} attack."
    config['experiment']['tags'] = ["federated", attack_type, "attack", "frequency", str(attack_frequency)]

    config['federated_learning']['attack_frequency'] = attack_frequency
    config['federated_learning']['attack_start_round'] = start_attack_round
    config['federated_learning']['attack_stop_round'] = stop_attack_round


def generate_fully_adv_attack_configs(base_config_path: str, attack_type: str, output_dir: str,
                                      datasets: List[str] = ['cifar10'], aggregation_names: List[str] = ['FedAvg']) -> \
List[str]:
    base_config = load_config(base_config_path)

    attack_type = attack_type.lower()
    attack_frequencies = [0]  # [0, -1, 10, 1] # 0: no attack, -1: random, 10: fixed frequency 10, 1: fixed frequency 1

    if attack_type not in ["base"]:
        attack_frequencies = [1]  # [-1, 1, 10] # 0: no attack, -1: random, 1: fixed frequency 1, 10: fixed frequency 10

    # Base setup for attacks
    if False:  # Placeholder for removed polymorph/iba/marksman attacks
        pass
    else:
        atk_eps_values = [0.03]
        target_labels = [0]
        atk_latent_dim = [-1]

    if aggregation_names[0] == 'all':
        aggregation_names = ["Median",
                             "CoordinateWiseMedian", "TrimmedMean", "RFA", "NormClipping", "WeakDP", "CRFL",
                             # "Krum", "MultiKrum", "Bulyan", "FoolsGold", "Flame", "DeepSight",
                             # "FLTrust", "FLDetector", "SimpleClustering",
                             ]
    else:
        # otherwise, test with the given aggregation methods
        aggregation_names = aggregation_names

    print(f"🚀 Testing with aggregation methods: {aggregation_names}")
    list_output_paths = []
    num_clients_list = [100]
    alpha_non_iid_list = [0.5]
    # alpha_non_iid_list = [0.1, 0.2, 1.0, 5.0, 10.0]
    # start_rounds = [400] # 500 0
    start_rounds = [0]  # 500 0
    num_rounds_training = [100]  # 600 # 200
    optimizers = ["Adam", "SGD"]
    check_optim_dataset = {
        "SGD": ['mnist', 'fashionmnist'],
        "Adam": ['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'gtsrb']
    }
    # BASE_CHECKPOINT_PAHTH = "./checkpoints/base/base_mnist_simplecnn_rnds_2000_opt_SGD__round_400.pth"
    datasets_name = ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet', 'gtsrb', 'svhn']
    models_name = ['simplecnn', 'simplecnn', 'resnet18', 'resnet18', 'resnet18', 'resnet18', 'resnet18']
    opt_name = ['SGD', 'Adam']
    # 'yaml': './configs/generated/base/base_svhn_resnet18_rnds_2000_opt_Adam.yaml',
    custom_models = {}
    for dataset, model in zip(datasets_name, models_name):
        custom_models[dataset] = {model: {}}
        for opt in opt_name:
            if dataset in check_optim_dataset[opt]:
                custom_models[dataset][model][
                    400] = f"./checkpoints/base/base_{dataset}_{model}_rnds_2000_opt_{opt}__round_400.pth"
                # custom_models[dataset][model][500] = f"./checkpoints/base/base_{dataset}_{model}_rnds_2000_opt_{opt}__round_500.pth"
                # custom_models[dataset][model][0] = ""
                custom_models[dataset][model][0] = "./checkpoints/clean_base/clean_base_cifar10_resnet18__round_300.pth"
                custom_models[dataset][model][
                    'yaml'] = f"./configs/generated/base/base_{dataset}_{model}_rnds_2000_opt_{opt}.yaml"

    # import json
    # print(json.dumps(custom_models, indent=4))
    # exit()

    for num_clients in num_clients_list:

        # number_adversarial_clients = 0 if attack_type in ["base"] else 4
        number_adversarial_clients = 10
        id_adversarial_clients = list(
            random.sample(range(num_clients), number_adversarial_clients)) if number_adversarial_clients > 0 else []
        print(
            f"🚀 Adversarial clients: {id_adversarial_clients} for {attack_type} attack with {number_adversarial_clients} adversarial clients")

        base_config['adversarial_clients'] = list(id_adversarial_clients)
        base_config['evaluation']['backdoor_evaluation'] = True
        base_config['federated_learning']['num_clients'] = num_clients

        base_config['experiment'][
            'description'] = f"Federated learning experiments evaluating the {attack_type.upper()} attack."
        base_config['experiment']['tags'] = ["federated", attack_type, "attack"]

        for alpha_non_iid in alpha_non_iid_list:
            for aggregation in aggregation_names:
                for dataset in datasets:
                    list_custom_models = custom_models[dataset].keys()
                    for model_name in list_custom_models:
                        DATASET_CONFIGS[dataset]['model_name'] = model_name
                        for start_round in start_rounds:
                            for nrd_training in num_rounds_training:
                                num_round = nrd_training + start_round
                                pretrained_model_path = custom_models[dataset][model_name][start_round]
                                for opt in optimizers:
                                    if start_round != 0:
                                        if opt.lower() not in pretrained_model_path.lower():
                                            continue

                                    if start_round == 0 and dataset not in check_optim_dataset[opt]:
                                        continue

                                    if "base" in attack_type:
                                        # for base, we don't need to set any attack configs
                                        config = clone_config(base_config)
                                        config['experiment'][
                                            'description'] = f"Federated learning experiments evaluating CLEAN training under different parameter settings."
                                        config['experiment']['tags'] = ["federated", "base", "clean"]

                                        # set federated learning configs
                                        config['federated_learning']['num_clients'] = num_clients
                                        config['federated_learning']['num_rounds'] = num_round
                                        config['adversarial_clients'] = []
                                        config['evaluation']['backdoor_evaluation'] = False

                                        config['model']['weights'] = pretrained_model_path

                                        set_dataset_normalization_and_optimizer(config, dataset, opt, alpha_non_iid)
                                        set_fl_aggregation(config, aggregation, dataset)
                                        suffix = f"{attack_type}_{dataset}_{model_name}_nc_{num_clients}_niid_{alpha_non_iid}_agg_{aggregation}_opt_{opt}_rnds_{num_round}_strnds_{start_round}"
                                        set_experiment_name(config, suffix, dataset)
                                        filename = f"{suffix}.yaml"
                                        output_path = write_config(config, output_dir, filename, attack_type)
                                        list_output_paths.append(output_path)
                                        continue

                                    for attack_frequency in attack_frequencies:
                                        for atk_eps in atk_eps_values:
                                            for flat_latent_dim in atk_latent_dim:
                                                for target_label in target_labels:
                                                    config = clone_config(base_config)

                                                    attack_duration = 100  # if start_round == 500 else 200 # 0, 400, 500
                                                    start_attack_round = start_round
                                                    stop_attack_round = start_attack_round + attack_duration
                                                    set_fl_attack_clients(config, attack_type, num_clients,
                                                                          start_attack_round, stop_attack_round,
                                                                          attack_frequency, num_round)

                                                    print(
                                                        f"🚀 Generating configs for: {dataset} with model: {model_name} aggregation: {aggregation} attack: {attack_type} and optimizer: {opt} and start_round: {start_round} end_round: {num_round} and attack_frequency: {attack_frequency} id_adversarial_clients: {id_adversarial_clients} atk_eps: {atk_eps} flat_latent_dim: {flat_latent_dim} target_label: {target_label}")

                                                    config['model']['weights'] = pretrained_model_path

                                                    set_dataset_normalization_and_optimizer(config, dataset, opt)
                                                    set_fl_aggregation(config, aggregation, dataset)

                                                    # if attack_type in ["badnets", "sinusoidal", "blended", "dba"]:
                                                    if attack_type not in ["base"]:
                                                        # pattern attack configs (badnets, sinusoidal, blended, dba) and base (clean training without attack)
                                                        set_pattern_attack_configs(config, dataset, attack_type,
                                                                                   target_label, id_adversarial_clients)
                                                    suffix = f"{attack_type}_{dataset}_{model_name}_nc_{num_clients}_niid_{alpha_non_iid}_agg_{aggregation}_opt_{opt}_rnds_{num_round}_strnds_{start_round}_nac_{number_adversarial_clients}_atkr_{start_attack_round}_stopr_{stop_attack_round}_atkf_{attack_frequency}_atk_eps_{atk_eps}_latent_dim_{flat_latent_dim}_label_{target_label}"
                                                    set_experiment_name(config, suffix, dataset)
                                                    filename = f"{suffix}.yaml"
                                                    output_path = write_config(config, output_dir, filename,
                                                                               attack_type)
                                                    list_output_paths.append(output_path)

    print(
        f"🎉 Type exps: {attack_type} -- Generated {len(list_output_paths)} config files in total for {attack_type} attack with aggregation {aggregation_names}")
    print(f"***" * 30)
    return list_output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate attack config files')
    parser.add_argument('--base', required=True, help='Base config file path')
    # parser.add_argument('--attack', nargs='+', choices=['base', 'sinusoidal', 'badnets', 'blended', 'dba'], required=True, help='Which attack(s) to generate (can specify multiple)')
    # parser.add_argument('--attack', nargs='+',
    #                     choices=['base', 'sinusoidal', 'badnets', 'blended', 'dba', 'neurotoxin', 'feddare',
    #                              'modelreplacement', 'threedfed', 'edgecase', 'labelflipping', 'layerwisepoisoning'],
    #                     required=True,
    #                     help='Which attack(s) to generate (can specify multiple)')
    parser.add_argument('--attack', nargs='+',
                        choices=['base', 'sinusoidal', 'badnets', 'blended', 'dba', 'neurotoxin', 'feddare',
                                 'modelreplacement', 'threedfed', 'edgecase', 'labelflipping', 'layerwisepoisoning',
                                 'minmax', 'trim', 'krum', 'cerp', 'a3fl', 'fcba', 'iba','darkfed'],  # 这里是新增的7种攻击
                        required=True,
                        help='Which attack(s) to generate (can specify multiple)')
    parser.add_argument('--output', default='configs/generated-v3', help='Output directory')
    parser.add_argument('--dataset', nargs='+', type=str, default=['cifar10'],
                        help='Dataset names for normalization and experiment naming')
    parser.add_argument('--aggregation', nargs='+', type=str, default=['FedAvg'],
                        help='Aggregation names for base')  # all means testing all aggregation methods, otherwise testing with FedAvg

    args = parser.parse_args()
    seed = 42
    set_random_seed(seed)

    print(f"🚀 Generating configs for: {', '.join(args.attack)}")
    print(f"📁 Output directory: {args.output}")
    print(f"📊 Datasets list: {', '.join(args.dataset)}")
    print(f"🔍 Aggregation methods: {', '.join(args.aggregation)}")

    total = 0
    path_configs = []

    for attack in args.attack:
        list_output_paths = generate_fully_adv_attack_configs(args.base, attack, args.output, args.dataset,
                                                              args.aggregation)
        path_configs.extend(list_output_paths)
        total += len(list_output_paths)

    print(f"\n🎉 Generated {total} config files in total!")

    # for ixd, path in enumerate(path_configs):
    #     print(f"{ixd+1}. {path}")
    # print("==="*30 + "\n")

    # Use relative path and standard venv activation
    project_root = os.path.dirname(os.path.abspath(__file__))
    # pattern_str = f'cd {project_root} && source .venv/bin/activate && stdbuf -oL -eL python -u run_federated.py --config configs/blended.yaml --gpu 7 2>&1 | stdbuf -oL -eL tee logs/blended.log'
    # pattern_str = f'cd {project_root} && source fl_env/bin/activate && stdbuf -oL -eL python -u run_federated.py --config configs/blended.yaml --gpu 7 2>&1 | stdbuf -oL -eL tee logs/blended.log'
    # ✅ 修改后的代码：彻底去掉虚拟环境激活命令，并默认将卡指定为 gpu 0
    pattern_str = f'cd {project_root} && stdbuf -oL -eL python -u run_federated.py --config configs/blended.yaml --gpu 0 2>&1 | stdbuf -oL -eL tee logs/blended.log'
    output_file = './commands-v3.txt'
    with open(output_file, 'a') as f:
        f.write(f"==" * 30 + '\n')
        f.write(f"# {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        # check all args and then combine to str 
        str_args = []
        for k, v in args.__dict__.items():
            if isinstance(v, list):
                str_args.append(f"--{k} {' '.join(v)}")
            else:
                str_args.append(f"--{k} {v}")
        args_str = f"Config: {' '.join(str_args)}"
        f.write(f"# {args_str}\n")
        for idx, path in enumerate(path_configs):
            pattern = pattern_str.replace('configs/blended.yaml', path)
            # ✅ 修改后的代码：对 8 取模，依次分配 0, 1, 2, 3, 4, 5, 6, 7
            # pattern = pattern.replace('gpu 7', f'gpu {str(int(idx) % 8)}')
            pattern = pattern.replace('gpu 7', 'gpu 0')
            log_file = os.path.basename(path).replace('.yaml', '')
            log_file = f'{log_file}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            attack_type = os.path.basename(path).split('_')[0].lower()
            os.makedirs(f'logs/{attack_type}', exist_ok=True)
            pattern = pattern.replace('logs/blended.log', f'logs/{attack_type}/{log_file}')
            print(f"{idx + 1}. {pattern}")
            f.write(pattern + '\n')
        f.write(f"**" * 30 + '\n')


if __name__ == '__main__':
    main()

# Example usage:
# python gen_exps_config.py --attack badnets sinusoidal blended dba --base configs/base.yaml --output configs/generated --dataset cifar10 --aggregation NormClipping WeakDP CenteredClipping CRFL
# python gen_exps_config.py --attack base --base configs/base.yaml --output configs/generated --dataset cifar10 cifar100 mnist fashionmnist svhn gtsrb tinyimagenet --aggregation FedAvg