"""
FL Server implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import pickle
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple
from datetime import datetime
from .memory import MemoryMonitor
from .aggregations import create_aggregation_method
from .defenses import create_defense
from .models import create_model_from_dataset_config
from .attacks import create_attack

class FLServer:
    """Enhanced FL Server with multiple aggregation methods"""

    def __init__(self, global_model: nn.Module, device: str, config: Dict):
        self.global_model = global_model
        self.device = device
        self.config = config

        # Checkpoint configuration
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        self.save_checkpoints = config['logging']['save_checkpoints']
        self.checkpoint_frequency = config['logging']['checkpoint_frequency']

        # Setup aggregation methods
        self.aggregation_methods = self._setup_aggregation_methods()
        self.server_defenses = self._setup_server_defenses()

        # 初始化 MAR 和 BAR 状态
        self.current_mar = 1.0
        self.current_bar = 1.0

    def _setup_aggregation_methods(self) -> List:
        """Setup aggregation methods"""
        methods = []
        aggregation_configs = self.config.get('server_aggregation', [])

        for agg_config in aggregation_configs:
            method = create_aggregation_method(agg_config)
            methods.append(method)

        return methods

    def _setup_server_defenses(self) -> List:
        """Setup server defenses"""
        defenses = []
        server_defenses = self.config.get('server_defenses', [])

        for defense_config in server_defenses:
            defense = create_defense(defense_config)
            defenses.append(defense)

        return defenses

    def aggregate_models(self, client_results: List[Dict], round_num: int) -> nn.Module:
        """Aggregate client models using selected method"""
        if not client_results:
            return self.global_model

        # =====================================================================
        # [实验 3.1] 特征提取探针：截取第 50 轮客户端最后一层更新 (Delta W)
        # =====================================================================
        TARGET_ROUND = 50
        if round_num == TARGET_ROUND:
            import os
            import torch
            import numpy as np
            os.makedirs("tsne_data", exist_ok=True)

            attack_name = self.config.get('attack', 'unknown').lower()
            global_state = self.global_model.state_dict()

            # 动态寻找分类层（最后一层权重），过滤掉偏置(bias)和批归一化(BN)层
            last_layer_key = None
            for key in reversed(list(global_state.keys())):
                if 'weight' in key and ('fc' in key or 'linear' in key or 'classifier' in key):
                    last_layer_key = key
                    break
            # 兜底：取模型参数字典的倒数第二个键
            if last_layer_key is None:
                last_layer_key = list(global_state.keys())[-2]

            print(f"\n[实验 3.1 探针] 截取特征轮次: {TARGET_ROUND} | 目标层: {last_layer_key}")

            X, y = [], []
            for res in client_results:
                client_state = res['model_state']
                if last_layer_key in client_state and last_layer_key in global_state:
                    # 严谨的张量处理：脱离计算图 -> 转CPU -> 转Numpy -> 展平
                    global_w = global_state[last_layer_key].detach().cpu()
                    client_w = client_state[last_layer_key].detach().cpu()
                    delta_w = (client_w - global_w).numpy().flatten()

                    X.append(delta_w)
                    # 标签标注：1=恶意更新, 0=良性更新
                    y.append(1 if res.get('active_attack', False) else 0)

            if len(X) > 0:
                save_path = f"tsne_data/tsne_{attack_name}.pt"
                torch.save({'X': np.array(X), 'y': np.array(y)}, save_path)
                print(f"[实验 3.1 成功] 截取 {len(X)} 个客户端特征，已存至 {save_path}\n")
            else:
                print("[实验 3.1 失败] 客户端权重解析异常，未提取到特征。")
        # =====================================================================

        # 下方保留原始的聚合逻辑，不作改动

        # Use first method for now
        agg_method = self.aggregation_methods[0]

        print(f"🔍 Server: aggregating models using {agg_method.__class__.__name__} with config: {agg_method.config}")

        # Apply server defenses before aggregation
        for defense in self.server_defenses:
            if defense.should_apply(round_num):
                # Convert client results to models for defense
                client_models = self._results_to_models(client_results)
                client_models = defense.apply(self.global_model, client_models, round_num)
                client_results = self._models_to_results(client_models, client_results)

        # Apply aggregation
        aggregated_model = agg_method.aggregate(self.global_model, client_results, round_num)

        # ======= 【新增代码】动态计算 MAR 和 BAR =======
        if hasattr(agg_method, 'last_accepted_clients'):
            accepted_clients = agg_method.last_accepted_clients
            adv_clients_config = self.config.get('adversarial_clients', [])

            # 统计当前轮次中，提交了更新的恶意/良性客户端总数
            adv_in_round = [r.get('client_id', -1) for r in client_results if r.get('client_id', -1) in adv_clients_config]
            benign_in_round = [r.get('client_id', -1) for r in client_results if r.get('client_id', -1) not in adv_clients_config]

            # 统计其中被防御算法“接受”的客户端数量
            accepted_adv = [c for c in accepted_clients if c in adv_in_round]
            accepted_benign = [c for c in accepted_clients if c in benign_in_round]

            # 计算百分比
            mar = (len(accepted_adv) / len(adv_in_round)) if adv_in_round else 0.0
            bar = (len(accepted_benign) / len(benign_in_round)) if benign_in_round else 0.0

            print(f"📊 Round {round_num} [{getattr(agg_method, 'name', 'Defense')}] | MAR: {mar*100:.1f}% | BAR: {bar*100:.1f}%")

            # 存入 server 对象，供 trainer 获取
            self.current_mar = mar
            self.current_bar = bar
        else:
            # 对于 FedAvg 等不筛选的算法，默认接受率为 100%
            self.current_mar = 1.0
            self.current_bar = 1.0
        # ==================================================

        # Move to CPU to save memory
        aggregated_model = aggregated_model.cpu()

        return aggregated_model

    def _results_to_models(self, client_results: List[Dict]) -> List[nn.Module]:
        """Convert client results to model instances"""
        models = []
        for result in client_results:
            model = create_model_from_dataset_config(self.config)
            model.load_state_dict(result['model_state'])
            models.append(model)
        return models

    def _models_to_results(self, models: List[nn.Module], original_results: List[Dict]) -> List[Dict]:
        """Convert model instances back to results format"""
        results = []
        for i, model in enumerate(models):
            result = original_results[i].copy()
            result['model_state'] = model.state_dict()
            results.append(result)
        return results

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate global model"""
        MemoryMonitor.monitor_memory("Before Evaluation", verbose=False)
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = F.cross_entropy(output, target)

                total_loss += loss.item() * data.size(0)
                correct += (output.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        self.global_model = self.global_model.cpu()
        MemoryMonitor.monitor_memory("After Evaluation", verbose=False)
        gc.collect()
        return accuracy, avg_loss, total_samples

    def _add_attack_config_norm(self, attack_config: Dict) -> Dict:
        """Add normalization values to attack config"""
        dataset_config = self.config['dataset']
        attack_config_with_norm = attack_config.copy()
        attack_config_with_norm['dataset_name'] = dataset_config['name']
        attack_config_with_norm['num_classes'] = dataset_config['num_classes']
        attack_config_with_norm['mean'] = dataset_config['mean']
        attack_config_with_norm['std'] = dataset_config['std']
        attack_config_with_norm['client_id'] = -1
        attack_config_with_norm['input_dim'] = dataset_config['data_shape'][1]
        attack_config_with_norm['attack_start_round'] = self.config['federated_learning']['attack_start_round']
        attack_config_with_norm['attack_stop_round'] = self.config['federated_learning']['attack_stop_round']
        attack_config_with_norm['attack_frequency'] = self.config['federated_learning']['attack_frequency']
        attack_config_with_norm['seed'] = self.config.get('experiment', {}).get('seed', 42)
        return attack_config_with_norm

    def evaluate_backdoor(self, test_loader: DataLoader, attack_config: Dict) -> Tuple[float, float]:
        """Evaluate global model with backdoor triggers"""
        MemoryMonitor.monitor_memory("Before Backdoor Evaluation", verbose=False)
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()

        attack_config_with_norm = self._add_attack_config_norm(attack_config)
        attack = create_attack(attack_config_with_norm)

        uses_random_target_label =  attack_config_with_norm.get('target_class', -1) == -1
        total_loss, correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                if uses_random_target_label:
                    data_filtered = data
                    labels_filtered = target
                else:
                    target_class = attack_config_with_norm.get('target_class', 0)
                    non_target_mask = (target != target_class)
                    if non_target_mask.sum() == 0:
                        continue
                    data_filtered = data[non_target_mask]
                    labels_filtered = target[non_target_mask]
                if len(data_filtered) == 0:
                    continue

                poisoned_data, poisoned_labels = attack.poison_data(data_filtered, labels_filtered)
                output = self.global_model(poisoned_data)
                loss = F.cross_entropy(output, poisoned_labels)

                predicted = output.argmax(dim=1)
                correct_predictions = (predicted == poisoned_labels).sum().item()

                total_loss += loss.item() * data_filtered.size(0)
                correct += correct_predictions
                total_samples += data_filtered.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        self.global_model = self.global_model.cpu()
        if attack.use_model_trigger():
            attack.atk_model = attack.atk_model.cpu()

        MemoryMonitor.monitor_memory("After Backdoor Evaluation", verbose=False)
        gc.collect()

        return accuracy, avg_loss, total_samples

    def save_checkpoint(self, round_num: int, metrics: Dict[str, Any], attack_models: Dict[str, Any] = None) -> str:
        """Save model checkpoint including generative models for attacks that require generators"""
        if not self.save_checkpoints:
            return None

        if round_num % self.checkpoint_frequency == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            experiment_name = self.config.get('experiment', {}).get('name', 'default_experiment')
            checkpoint_path = f"{self.checkpoint_dir}/{experiment_name}__round_{round_num}.pth"

            model_cpu = self.global_model.cpu()

            checkpoint_data = {
                'round_num': round_num,
                'model_state_dict': model_cpu.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
            }

            if attack_models is None:
                attack_models = self._collect_attack_models_from_caches()

            if attack_models:
                attack_models_cpu = {}
                for attack_name, attack_data in attack_models.items():
                    model = attack_data.get('model')
                    if model is not None:
                        model_cpu = model.cpu()
                        attack_models_cpu[attack_name] = {
                            'model_state_dict': model_cpu.state_dict(),
                            'attack_config': attack_data['config'],
                            'attack_class': attack_data['attack_class']
                        }

                if attack_models_cpu:
                    checkpoint_data['attack_models'] = attack_models_cpu
                del attack_models

            torch.save(checkpoint_data, checkpoint_path)
            del checkpoint_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return checkpoint_path

        return None

    def _collect_attack_models_from_caches(self) -> Dict[str, Any]:
        return {}

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()