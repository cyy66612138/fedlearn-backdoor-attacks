"""
FL Client implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Any
from .memory import MemoryMonitor
from .attacks import create_attack
from .defenses import create_defense
from .models import create_model_from_dataset_config
import time
import inspect

class FLClient:
    """Enhanced FL Client with multiple attacks/defenses support"""

    def __init__(self, client_id: int, dataset: Subset, device: str, config: Dict):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.config = config
        self.model = None
        self.optimizer = None
        self.dataset_name = self.config['dataset']['name']
        self.mean = self.config['dataset']['mean']
        self.std = self.config['dataset']['std']
        self.num_classes = self.config['dataset']['num_classes']
        self.data_shape = self.config['dataset']['data_shape']
        self.input_dim = self.data_shape[1]

        self.adversarial_client_ids = self.config.get('adversarial_clients', [])
        self.attacks = self._setup_attacks()
        self.defenses = self._setup_defenses()

    def _setup_attacks(self) -> List:
        attacks = []
        client_attacks = self.config.get('client_attacks', [])
        for attack_config in client_attacks:
            apply_to_client_ids = attack_config['apply_to_client_ids']
            if self.client_id in apply_to_client_ids and self.client_id in self.adversarial_client_ids:
                attack_config_with_norm = attack_config.copy()
                attack_config_with_norm['mean'] = self.mean
                attack_config_with_norm['std'] = self.std
                attack_config_with_norm['dataset_name'] = self.dataset_name
                attack_config_with_norm['num_classes'] = self.num_classes
                attack_config_with_norm['input_dim'] = self.input_dim
                attack_config_with_norm['client_id'] = self.client_id
                attack_config_with_norm['attack_start_round'] = self.config['federated_learning']['attack_start_round']
                attack_config_with_norm['attack_stop_round'] = self.config['federated_learning']['attack_stop_round']
                attack_config_with_norm['attack_frequency'] = self.config['federated_learning']['attack_frequency']
                attack_config_with_norm['seed'] = self.config.get('experiment', {}).get('seed', 42)
                attack = create_attack(attack_config_with_norm)
                attacks.append(attack)
        return attacks

    def load_attack_models(self, attack_models_data: Dict[str, Any]) -> None:
        if not attack_models_data or self.client_id not in self.adversarial_client_ids:
            return
        for attack_name, attack_data in attack_models_data.items():
            try:
                for attack in self.attacks:
                    if attack.__class__.__name__ == attack_data.get('attack_class'):
                        if hasattr(attack, 'atk_model') and attack.atk_model is not None:
                            attack.atk_model.load_state_dict(attack_data['model_state_dict'])
                            attack.atk_model.eval()
                            print(f"Set up attack model for client {self.client_id}: Loaded {attack_name} model")
            except Exception as e:
                print(f"Client {self.client_id}: Failed to load {attack_name}: {str(e)}")

    def _setup_defenses(self) -> List:
        defenses = []
        client_defenses = self.config.get('client_defenses', [])
        for defense_config in client_defenses:
            apply_to_client_ids = defense_config.get('apply_to_client_ids')
            if self.client_id in apply_to_client_ids:
                defense = create_defense(defense_config)
                defenses.append(defense)
        return defenses

    def set_model(self, global_model: nn.Module):
        self.model = create_model_from_dataset_config(self.config)
        with torch.no_grad():
            self.model.load_state_dict(global_model.state_dict())
            self.global_model_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}

        optimizer = self.config['federated_learning'].get('optimizer', 'SGD')
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
            lr=self.config['federated_learning']['learning_rate'],
            momentum=self.config['federated_learning']['momentum'],
            weight_decay=self.config['federated_learning']['weight_decay'])
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),
            lr=self.config['federated_learning']['learning_rate'],
            weight_decay=self.config['federated_learning']['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _has_active_attack_this_round(self, round_idx):
        if self.client_id not in self.adversarial_client_ids:
            return False
        for attack in self.attacks:
            apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
            if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
                return True
        return False

    def train(self, epochs: int = 2, batch_size: int = 128, round_idx: int = 0, base_seed: int = 42) -> Dict:
        start_time = time.time()
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        print(f"*"*50)
        self.model = self.model.to(self.device)
        self.model.train()

        MemoryMonitor.monitor_memory(f"Client {self.client_id} - Training Start", verbose=False)
        client_seed = base_seed + self.client_id * 1000 + round_idx
        local_generator = torch.Generator().manual_seed(client_seed)
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False, generator=local_generator
        )

        has_active_attack = self._has_active_attack_this_round(round_idx)
        epochs_to_run = epochs

        if has_active_attack:
            epochs_to_run = self.config.get('adversarial_epochs', 6)
            for attack in self.attacks:
                if hasattr(attack, 'train_attack_model'):
                    attack.train_attack_model(self.model, dataloader, self.client_id, self.device, verbose=True)

        # =========================================================================
        # 💥 核心逻辑：双轨训练 (Dual-Track Training) 💥
        # 为了严格获取 G_benign 用于 FedDARE 方案一的掩码计算，进行良性预演
        # =========================================================================
        benign_model_state_for_attack = None
        needs_dual_track = False
        if has_active_attack:
            for attack in self.attacks:
                if attack.__class__.__name__ == 'FedDAREAttack' and attack.should_apply(round_idx):
                    needs_dual_track = True
                    break

        if needs_dual_track:
            print(f"   [Dual-Track] Client {self.client_id} starting Benign pre-run for FedDARE...")
            initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # 第一轨：纯净数据训练
            for epoch in range(epochs_to_run):
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    if data.size(0) == 1: continue
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.optimizer.step()

            # 保存真实良性分布
            benign_model_state_for_attack = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 模型回滚，重置优化器
            self.model.load_state_dict(initial_state)
            opt_name = self.config['federated_learning'].get('optimizer', 'SGD')
            if opt_name == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(),
                    lr=self.config['federated_learning']['learning_rate'],
                    momentum=self.config['federated_learning']['momentum'],
                    weight_decay=self.config['federated_learning']['weight_decay'])
            elif opt_name == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(),
                    lr=self.config['federated_learning']['learning_rate'],
                    weight_decay=self.config['federated_learning']['weight_decay'])
            print(f"   [Dual-Track] Benign pre-run completed. Starting actual Poison run...")
        # =========================================================================

        # 常规的第二轨训练：带有数据投毒 (Poison Run)
        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        start_time = time.time()
        for epoch in range(epochs_to_run):
            epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                if data.size(0) == 1: continue

                if has_active_attack:
                    for attack in self.attacks:
                        apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
                        if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
                            data, target = attack.poison_data(data, target)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_correct += (output.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

                if batch_idx % 200 == 0:
                    MemoryMonitor.cleanup_memory()

            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples
            if epoch == epochs_to_run - 1:
                print(f"   Client {self.client_id}, Epoch {epoch+1}/{epochs_to_run}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Time: {time.time() - start_time:.2f}s")

        with torch.no_grad():
            model_cpu = self.model.cpu()
            model_state = model_cpu.state_dict()
            del model_cpu

            if has_active_attack:
                for attack in self.attacks:
                    apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
                    if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
                        if hasattr(attack, 'setup_lsa_environment'):
                            attack.setup_lsa_environment(model=self.model, dataloader=dataloader, device=self.device)

                        agg_algo = self.config.get('server_aggregation', [{'name': 'FedAvg'}])[0]['name']

                        # 动态兼容机制：如果攻击类支持接收 benign_model_state 则传入
                        sig = inspect.signature(attack.apply_model_poisoning)
                        if 'benign_model_state' in sig.parameters:
                            model_state = attack.apply_model_poisoning(
                                local_model_state=model_state,
                                global_model_state=self.global_model_state,
                                benign_model_state=benign_model_state_for_attack,
                                algorithm=agg_algo
                            )
                        else:
                            model_state = attack.apply_model_poisoning(
                                local_model_state=model_state,
                                global_model_state=self.global_model_state,
                                algorithm=agg_algo
                            )

        result = {
            'model_state': model_state,
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'samples': epoch_samples,
            'active_attack': has_active_attack,
            'client_id': self.client_id,
            'round': round_idx,
        }

        self.cleanup_memory()
        return result

    def cleanup_memory(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None

        if hasattr(self, 'global_model_state'):
            del self.global_model_state

        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()