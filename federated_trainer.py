"""
Experiment Runner for FL Research Framework (Modular Version)
Concise version using core modular structure
"""

import numpy as np
import random
import gc
from typing import Dict, List, Any
from datetime import datetime
import os
import yaml
import wandb
import json
import time
import torch  # 确保引入 torch 用于创建 Subset

from core import FLResearchFramework, MemoryMonitor

class FederatedTrainer:
    """Concise experiment runner using modular core structure"""

    def __init__(self, config: Dict[str, Any], framework: FLResearchFramework, data_loader):
        self.config = config
        self.framework = framework
        self.data_loader = data_loader
        self.training_history = []
        self.wandb_run = self.config['logging']['use_wandb']

    def run_training(self) -> Dict[str, Any]:
        """Run complete FL training experiment"""
        print("\n🎯 Starting FL Training Experiment")
        print("=" * 60)

        # Setup experiment
        _, _, client_datasets = self.data_loader.load_datasets()
        test_loader = self.data_loader.get_test_loader()
        checkpoint_path = self.config['model'].get('weights', None)
        if checkpoint_path:
            print(f"🔍 Attempting to load checkpoint: {checkpoint_path}")
        else:
            print(f"🔄 No checkpoint specified, starting from scratch")

        start_round = self.framework.setup_experiment(None, test_loader, client_datasets, checkpoint_path)

        # 检查是否仅初始化权重而不继承保存的轮次
        resume_training = self.config['model'].get('resume_training', False)
        if checkpoint_path and not resume_training:
            print(f"\n💡 Only initializing model weights from checkpoint. Round counter reset to 0.")
            print(
                f"   (Tip: set `resume_training: true` in your yaml under `model` if you want to continue from the saved round)\n")
            start_round = 0

        print(f"📌 Starting training from round {start_round}")

        # =====================================================================
        # 🟢 【终极修复】为 FLARE / FLTrust 动态注入缩小版的 central_dataset (防卡死)
        # =====================================================================
        if hasattr(self.framework, 'server') and hasattr(self.framework.server, 'aggregation_methods'):
            for agg_method in self.framework.server.aggregation_methods:
                if agg_method.name in ['FLARE', 'FLTrust', 'flare'] or 'FLARE' in agg_method.__class__.__name__:
                    # 抽取 500 个样本作为中央数据集
                    subset_size = 50
                    total_size = len(test_loader.dataset)
                    actual_size = min(subset_size, total_size)
                    subset_indices = list(range(actual_size))
                    small_central_dataset = torch.utils.data.Subset(test_loader.dataset, subset_indices)

                    print(f"💉 [Hotfix] Injecting a subset ({actual_size} samples) as central_dataset for {agg_method.name} to prevent hanging...")

                    if not hasattr(agg_method, 'params'):
                        agg_method.params = {}
                    agg_method.params['central_dataset'] = small_central_dataset
        # =====================================================================

        print("=" * 60)
        # Initial evaluation
        log_metrics = self._evaluate_model(test_loader, "Initial")
        if self.wandb_run:
            wandb.log(log_metrics, step=start_round)

        print("=" * 60)

        # Training parameters
        num_rounds = self.config['federated_learning']['num_rounds']
        num_clients_per_round = int(self.config['federated_learning']['num_clients'] * self.config['federated_learning']['participation_rate'])

        MemoryMonitor.monitor_memory("Training Start")

        # Training loop
        for round_idx in range(start_round, num_rounds):
            print("=" * 60)
            print(f"\n🔄 ROUND {round_idx + 1}/{num_rounds}")

            # Run training round
            MemoryMonitor.reset_peaks()
            selected_clients = self._select_clients(round_idx, num_clients_per_round)
            round_metrics = self.framework.run_training_round(selected_clients, round_idx)
            # Evaluation and logging
            self._evaluate_round(round_idx, test_loader, round_metrics)
            self._save_checkpoint(round_idx, round_metrics)
            self._log_round(round_idx, round_metrics)
            MemoryMonitor.cleanup_memory(aggressive=True)

        # Final results
        final_accuracy, final_loss, final_samples = self.framework.evaluate(test_loader)
        print(f"\n🏁 Final Test Accuracy: {final_accuracy:.4f}, Final Test Loss: {final_loss:.4f}")

        if self.wandb_run:
            self._log_final_metrics(final_accuracy, final_loss, final_samples, num_rounds)

        return self._prepare_results(final_accuracy, final_loss, final_samples, num_rounds)

    def _evaluate_model(self, test_loader, phase: str):
        """Evaluate model and print results"""
        log_metrics = {}
        test_accuracy, test_loss, test_samples = self.framework.evaluate(test_loader)
        log_metrics.update({
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'test_samples': int(test_samples)
        })
        print(f"🔍 {phase} Main Accuracy: {test_accuracy:.4f}, Main Loss: {test_loss:.4f}, Main Samples: {test_samples}")
        for id_attack, attack_config in enumerate(self.config.get('client_attacks', [])):
            backdoor_accuracy, backdoor_loss, backdoor_samples = self.framework.evaluate_backdoor(test_loader, attack_config)
            print(f"🔍 {phase} Backdoor Accuracy (attack {id_attack}): {backdoor_accuracy:.4f}, Backdoor Loss: {backdoor_loss:.4f}, Backdoor Samples: {backdoor_samples}")
            log_metrics.update({
                f'backdoor_accuracy_{id_attack}': float(backdoor_accuracy),
                f'backdoor_loss_{id_attack}': float(backdoor_loss),
                f'backdoor_samples_{id_attack}': int(backdoor_samples),
            })
        return log_metrics

    def _evaluate_round(self, round_idx: int, test_loader, round_metrics: Dict):
        """Evaluate model for current round"""
        if (round_idx + 1) % self.config['evaluation']['test_frequency'] == 0:
            start_time = time.time()
            test_accuracy, test_loss, test_samples = self.framework.evaluate(test_loader)
            round_metrics.update({
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'test_samples': int(test_samples),
                'test_time': time.time() - start_time
            })

            # Backdoor evaluation
            if self.config['evaluation'].get('backdoor_evaluation', False):
                backdoor_start_time = time.time()
                for id_attack, attack_config in enumerate(self.config.get('client_attacks', [])):
                    backdoor_accuracy, backdoor_loss, backdoor_samples = self.framework.evaluate_backdoor(test_loader, attack_config)
                    round_metrics.update({
                        f'backdoor_accuracy_{id_attack}': float(backdoor_accuracy),
                        f'backdoor_loss_{id_attack}': float(backdoor_loss),
                        f'backdoor_samples_{id_attack}': int(backdoor_samples),

                    })
                round_metrics.update({
                    'backdoor_time': time.time() - backdoor_start_time
                })
        else:
            round_metrics.update({
                'test_accuracy': 0.0, 'test_loss': 0.0, 'test_samples': 0, 'test_time': 0,
                'backdoor_accuracy_0': 0.0, 'backdoor_loss_0': 0.0, 'backdoor_samples_0': 0, 'backdoor_time': 0
            })


    def _save_checkpoint(self, round_idx: int, round_metrics: Dict):
        """Save checkpoint if needed"""
        checkpoint_path = self.framework.server.save_checkpoint(round_idx + 1, round_metrics)
        if checkpoint_path:
            round_metrics['checkpoint_path'] = checkpoint_path

    def _log_round(self, round_idx: int, round_metrics: Dict):
        """Log round metrics"""
        self._print_round_stats(round_metrics)

        if self.wandb_run:
            log_metrics = round_metrics.copy()
            if 'client_training_times' in log_metrics:
                del log_metrics['client_training_times']
            wandb.log(log_metrics, step=round_idx + 1)

        self.training_history.append(round_metrics)

    # def _select_clients(self, round_idx: int, num_clients_per_round: int) -> List:
    #     """Select clients for current round"""
    #     all_clients = self.framework.clients
    #     attack_frequency = self.config['federated_learning'].get('attack_frequency', 0)
    #
    #     if attack_frequency > 0:
    #         attack_start_round = self.config['federated_learning']['attack_start_round']
    #         attack_stop_round = self.config['federated_learning']['attack_stop_round']
    #
    #         if (round_idx - attack_start_round) % attack_frequency == 0 and round_idx < attack_stop_round:
    #             adversarial_ids = self.config.get('adversarial_clients', [])
    #             adversarial_clients = [c for c in all_clients if c.client_id in adversarial_ids]
    #             benign_clients = [c for c in all_clients if c.client_id not in adversarial_ids]
    #
    #             selected = adversarial_clients.copy()
    #             remaining = num_clients_per_round - len(adversarial_clients)
    #             if remaining > 0:
    #                 selected.extend(random.sample(benign_clients, min(remaining, len(benign_clients))))
    #             return selected
    #         else:
    #             return random.sample(all_clients, num_clients_per_round)
    #     elif attack_frequency == -1:
    #         return random.sample(all_clients, num_clients_per_round)
    #     elif attack_frequency == 0:
    #         return random.sample(all_clients, num_clients_per_round)
    #     else:
    #         raise ValueError(f"Unknown attack frequency: {attack_frequency}")
    def _select_clients(self, round_idx: int, num_clients_per_round: int) -> List:
        """Select clients for current round"""
        all_clients = self.framework.clients
        attack_frequency = self.config['federated_learning'].get('attack_frequency', 0)

        if attack_frequency > 0:
            attack_start_round = self.config['federated_learning']['attack_start_round']
            attack_stop_round = self.config['federated_learning']['attack_stop_round']

            if (round_idx - attack_start_round) % attack_frequency == 0 and round_idx < attack_stop_round:
                adversarial_ids = self.config.get('adversarial_clients', [])

                # 分离恶意和良性池
                adversarial_pool = [c for c in all_clients if c.client_id in adversarial_ids]
                benign_pool = [c for c in all_clients if c.client_id not in adversarial_ids]

                selected = []
                # 🎯 核心修正：只从恶意池中随机挑 1 个！
                if adversarial_pool:
                    selected.append(random.choice(adversarial_pool))

                # 补齐剩下的良性客户端 (比如 9 个)
                remaining = num_clients_per_round - len(selected)
                if remaining > 0 and benign_pool:
                    selected.extend(random.sample(benign_pool, min(remaining, len(benign_pool))))

                return selected
            else:
                return random.sample(all_clients, num_clients_per_round)
        elif attack_frequency == -1 or attack_frequency == 0:
            return random.sample(all_clients, num_clients_per_round)
        else:
            raise ValueError(f"Unknown attack frequency: {attack_frequency}")

    def _print_round_stats(self, round_metrics: Dict[str, Any]):
        """Print round statistics"""
        r = round_metrics
        client_times = f"[{', '.join([f'{t:.2f}' for t in r['client_training_times']])}]"

        print(f"📊 Round {r['round']}: "
              f"Train Acc={r['train_accuracy']:.4f}, Loss={r['train_loss']:.4f}, "
              f"Samples={r['total_samples']}, CPU={r['peak_cpu_memory_gb']:.2f}GB, "
              f"GPU={r['peak_gpu_memory_gb']:.3f}GB")

        print(f"\tTest: Acc={r.get('test_accuracy', 0.0):.4f}, "
              f"Loss={r.get('test_loss', 0.0):.4f}, "
              f"Samples={r.get('test_samples', 0)}, "
              f"Time={r.get('test_time', 0.0):.2f}s")

        for id_attack, attack_config in enumerate(self.config.get('client_attacks', [])):
            print(f"\tBackdoor {id_attack}: Acc={r.get(f'backdoor_accuracy_{id_attack}', 0.0):.4f}, "
                  f"Loss={r.get(f'backdoor_loss_{id_attack}', 0.0):.4f}, "
                  f"Samples={r.get(f'backdoor_samples_{id_attack}', 0)}, "
                  f"Time={r.get(f'backdoor_time', 0.0):.2f}s")

        print(f"\tTiming: Total={r['total_round_time_seconds']:.2f}s, "
              f"Min={r['minimal_time_seconds']:.2f}s, "
              f"Dist={r['distribute_time_seconds']:.2f}s, "
              f"Client={client_times}, "
              f"Agg={r['aggregation_time_seconds']:.2f}s")

    def _log_final_metrics(self, final_accuracy: float, final_loss: float, final_samples: int, num_rounds: int):
        """Log final metrics to wandb"""
        if not self.wandb_run:
            return
        train_accs = [r['train_accuracy'] for r in self.training_history]
        train_losses = [r['train_loss'] for r in self.training_history]

        final_metrics = {
            'final_test_accuracy': float(final_accuracy),
            'final_test_loss': float(final_loss),
            'final_test_samples': int(final_samples),
            'best_train_accuracy': float(np.max(train_accs)),
            'avg_train_accuracy': float(np.mean(train_accs)),
            'best_train_loss': float(np.min(train_losses)),
            'avg_train_loss': float(np.mean(train_losses))
        }
        wandb.log(final_metrics)
        wandb.summary.update({
            'final_test_accuracy': float(final_accuracy),
            'final_test_samples': int(final_samples),
            'best_train_accuracy': float(np.max(train_accs))
        })

    # def _prepare_results(self, final_accuracy: float, final_loss: float, final_samples: int, num_rounds: int) -> Dict[str, Any]:
    #     """Prepare final results"""
    #     return {
    #         'config': self.config,
    #         'training_history': self.training_history,
    #         'final_accuracy': float(final_accuracy),
    #         'final_loss': float(final_loss),
    #         'final_samples': int(final_samples),
    #         'dataset_info': self.data_loader.get_dataset_info(),
    #         'experiment_metadata': {
    #             'end_time': datetime.now().isoformat(),
    #             'total_rounds': num_rounds,
    #             'num_clients': self.config['federated_learning']['num_clients']
    #         }
    #     }
    def _prepare_results(self, final_accuracy: float, final_loss: float, final_samples: int, num_rounds: int) -> Dict[
        str, Any]:
        """Prepare final results"""

        # =====================================================================
        # 🟢 【JSON 序列化修复】清理掉 config 中不能存进 JSON 的 PyTorch 数据集对象
        # =====================================================================
        def clean_config_dict(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == 'central_dataset':
                        d[k] = "PyTorch Subset (Removed to fix JSON dump)"
                    else:
                        clean_config_dict(v)
            elif isinstance(d, list):
                for item in d:
                    clean_config_dict(item)

        # 递归清理 config 里的不可序列化对象
        clean_config_dict(self.config)
        # =====================================================================

        return {
            'config': self.config,
            'training_history': self.training_history,
            'final_accuracy': float(final_accuracy),
            'final_loss': float(final_loss),
            'final_samples': int(final_samples),
            'dataset_info': self.data_loader.get_dataset_info(),
            'experiment_metadata': {
                'end_time': datetime.now().isoformat(),
                'total_rounds': num_rounds,
                'num_clients': self.config['federated_learning']['num_clients']
            }
        }

    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = self.config['logging']['save_results_dir']

        yaml_path = f"{folder_name}/{filename}_{timestamp}.yaml"
        json_path = f"{folder_name}/{filename}_{timestamp}.json"
        os.makedirs(folder_name, exist_ok=True)

        # Save as YAML
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)

        # Save as JSON
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        return yaml_path, json_path