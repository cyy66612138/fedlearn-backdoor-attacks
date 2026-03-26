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
                attack = create_attack(attack_config_with_norm)
                attacks.append(attack)
        return attacks

    def _setup_defenses(self) -> List:
        return []

    def set_model(self, global_model: nn.Module):
        self.model = create_model_from_dataset_config(self.config)
        with torch.no_grad():
            self.model.load_state_dict(global_model.state_dict())
            self.global_model_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}

        optimizer = self.config['federated_learning'].get('optimizer', 'SGD')
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['federated_learning']['learning_rate'], momentum=self.config['federated_learning']['momentum'], weight_decay=self.config['federated_learning']['weight_decay'])
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['federated_learning']['learning_rate'], weight_decay=self.config['federated_learning']['weight_decay'])

    def _has_active_attack_this_round(self, round_idx):
        if self.client_id not in self.adversarial_client_ids:
            return False
        for attack in self.attacks:
            apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
            if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
                return True
        return False

    def train(self, epochs: int = 2, batch_size: int = 128, round_idx: int = 0, base_seed: int = 42) -> Dict:
        if self.model is None: raise ValueError("Model not set.")
        self.model = self.model.to(self.device)
        self.model.train()

        client_seed = base_seed + self.client_id * 1000 + round_idx
        local_generator = torch.Generator().manual_seed(client_seed)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, generator=local_generator)

        has_active_attack = self._has_active_attack_this_round(round_idx)

        # --- 动态调整 Epoch 逻辑 ---
        if has_active_attack:
            # 如果是恶意客户端，在此处硬编码或从配置读取倍率
            # 建议设置为 5 或者 epochs * 2.5
            epochs_to_run = 6
            print(f"   [🔥 Attack Boost] Client {self.client_id} is attacking, boosting local epochs to {epochs_to_run}")
        else:
            # 良性客户端维持原状
            epochs_to_run = epochs

        # 1. 干净的预演轨 (提取良性基底)
        benign_model_state_for_attack = None

        # [修改点] 将 LayerwisePoisoningAttack 也加入到需要跑 benign track 的判断条件中
        needs_benign_track = has_active_attack and any(
            (a.__class__.__name__ in ['FedDAREAttack', 'LayerwisePoisoningAttack']) and a.should_apply(round_idx)
            for a in self.attacks
        )

        if needs_benign_track:
            print(f"   [Benign Track] Client {self.client_id} running benign track for mask proxy...")
            initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            for epoch in range(epochs_to_run):
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(self.device), target.to(self.device)
                    if data.size(0) == 1: continue
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    self.optimizer.step()
            benign_model_state_for_attack = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(initial_state)

            opt_name = self.config['federated_learning'].get('optimizer', 'SGD')
            if opt_name == "SGD": self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['federated_learning']['learning_rate'], momentum=self.config['federated_learning']['momentum'], weight_decay=self.config['federated_learning']['weight_decay'])
            elif opt_name == "Adam": self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['federated_learning']['learning_rate'], weight_decay=self.config['federated_learning']['weight_decay'])

        # 2. 纯正的带毒训练 (无任何限制)
        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        start_time = time.time()
        for epoch in range(epochs_to_run):
            epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                if data.size(0) == 1: continue

                if has_active_attack:
                    for attack in self.attacks:
                        if attack.should_apply(round_idx):
                            data, target = attack.poison_data(data, target)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_correct += (output.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            if epoch == epochs_to_run - 1:
                print(f"   Client {self.client_id}, Epoch {epoch+1}/{epochs_to_run}, Loss: {epoch_loss/epoch_samples:.4f}, Acc: {epoch_correct/epoch_samples:.4f}, Time: {time.time() - start_time:.2f}s")

        # [新增核心逻辑] 为需要 LSA (Layer Substitution) 的攻击挂载本地模型和测试数据环境
        if has_active_attack:
            for attack in self.attacks:
                if attack.should_apply(round_idx) and hasattr(attack, 'setup_lsa_environment'):
                    # 传入不需要 shuffle 的 dataloader 以保证 LSA 测试的前后对比一致性
                    lsa_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
                    attack.setup_lsa_environment(self.model, lsa_loader, self.device)

        # 3. 提交给 attacks.py 组装
        with torch.no_grad():
            model_cpu = self.model.cpu()
            model_state = model_cpu.state_dict()
            del model_cpu

            if has_active_attack:
                for attack in self.attacks:
                    if attack.should_apply(round_idx):
                        agg_algo = self.config.get('server_aggregation', [{'name': 'FedAvg'}])[0]['name']
                        sig = inspect.signature(attack.apply_model_poisoning)
                        if 'benign_model_state' in sig.parameters:
                            model_state = attack.apply_model_poisoning(
                                local_model_state=model_state, global_model_state=self.global_model_state,
                                benign_model_state=benign_model_state_for_attack, algorithm=agg_algo)
                        else:
                            model_state = attack.apply_model_poisoning(
                                local_model_state=model_state, global_model_state=self.global_model_state, algorithm=agg_algo)

        result = {'model_state': model_state, 'loss': epoch_loss/epoch_samples, 'accuracy': epoch_correct/epoch_samples, 'samples': epoch_samples, 'active_attack': has_active_attack, 'client_id': self.client_id, 'round': round_idx}
        self.cleanup_memory()
        return result

    def cleanup_memory(self):
        if self.model is not None: self.model.cpu(); del self.model; self.model = None
        if hasattr(self, 'global_model_state'): del self.global_model_state
        if self.optimizer is not None: del self.optimizer; self.optimizer = None
        if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()