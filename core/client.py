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
# Removed global_generator import to avoid circular import
    
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
        self.data_shape = self.config['dataset']['data_shape'] # [C, H, W]
        self.input_dim = self.data_shape[1]
        
        # Attack and defense configurations
        self.adversarial_client_ids = self.config.get('adversarial_clients', [])
        self.attacks = self._setup_attacks()
        self.defenses = self._setup_defenses()
    
    def _setup_attacks(self) -> List:
        """Setup attacks for this client"""
        attacks = []
        client_attacks = self.config.get('client_attacks', [])
        # print(f"🔍 Client {self.client_id}: setup attacks with config: {self.config}")
        for attack_config in client_attacks:
            apply_to_client_ids = attack_config['apply_to_client_ids']
            # print(f"🔍 Client {self.client_id}: apply_to_client_ids: {apply_to_client_ids}")
            # print(f"🔍 Client {self.client_id}: adversarial_client_ids: {self.adversarial_client_ids}")
            # import IPython; IPython.embed(); exit()
            # Check if attack should be applied to this client
            if self.client_id in apply_to_client_ids and self.client_id in self.adversarial_client_ids:
                # Add dataset normalization values to attack config
                attack_config_with_norm = attack_config.copy()
                attack_config_with_norm['mean'] = self.mean
                attack_config_with_norm['std'] = self.std
                # checking if any attribute is missing or redundant
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
        """Load attack models from checkpoint data"""
        if not attack_models_data or self.client_id not in self.adversarial_client_ids:
            return
        # Duplicate loading of attack models when have multiple attackers
        # might change it to non-cooperative attack
        for attack_name, attack_data in attack_models_data.items():
            try:
                # Find matching attack in this client
                for attack in self.attacks:
                    # print(f"🔍 Client {self.client_id}: Loading {attack_name} model attack class name: {attack_data.get('attack_class')} and attack class: {attack.__class__.__name__}")
                    if attack.__class__.__name__ == attack_data.get('attack_class'):
                        # Load the generative model state dict
                        if hasattr(attack, 'atk_model') and attack.atk_model is not None:
                            attack.atk_model.load_state_dict(attack_data['model_state_dict'])
                            # attack.atk_model = attack.atk_model.to(self.device)
                            attack.atk_model.eval()
                            print(f"Set up attack model for client {self.client_id}: Loaded {attack_name} model")
                        # break
            except Exception as e:
                print(f"Client {self.client_id}: Failed to load {attack_name}: {str(e)}")
    
    def _setup_defenses(self) -> List:
        """Setup defenses for this client"""
        defenses = []
        client_defenses = self.config.get('client_defenses', [])
        
        for defense_config in client_defenses:
            apply_to_client_ids = defense_config.get('apply_to_client_ids')
            if self.client_id in apply_to_client_ids:
                defense = create_defense(defense_config)
                defenses.append(defense)
        
        return defenses
    
    
    def set_model(self, global_model: nn.Module):
        """Set global model for local training"""
        self.model = create_model_from_dataset_config(self.config)
        
        # Load global model weights
        with torch.no_grad():
            self.model.load_state_dict(global_model.state_dict())
            # === 新增：保存一份全局模型的副本（CPU上），供后续模型投毒对比使用 ===
            self.global_model_state = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
            # self.model = self.model.to(self.device)
        
        # print l2 norm of the global model and local model
        # global_model_norm = torch.norm(torch.cat([p.view(-1) for p in global_model.parameters()]), p=2).item()
        # model_norm = torch.norm(torch.cat([p.view(-1) for p in self.model.parameters()]), p=2).item()
        # print(f"🔍 Client {self.client_id} get distributed model from global_model_norm: {global_model_norm} local_model_norm: {model_norm}")

        # Setup optimizer
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
        """Check if this client has any active attacks for the current round"""
        if self.client_id not in self.adversarial_client_ids:
            return False
        
        for attack in self.attacks:
            apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
            if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
                return True
        return False
        
    def train(self, epochs: int = 2, batch_size: int = 128, round_idx: int = 0, base_seed: int = 42) -> Dict:
        """Local training with attacks and defenses"""
        start_time = time.time()
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        print(f"*"*50)
        # # print l2 norm of the model
        # model_params = list(self.model.parameters())
        # model_norm = torch.norm(torch.cat([p.view(-1) for p in model_params]), p=2).item()
        # print(f"In train() function of client {self.client_id}: model_norm: {model_norm}")
        
        self.model = self.model.to(self.device)
        self.model.train()

        # Memory monitoring
        MemoryMonitor.monitor_memory(f"Client {self.client_id} - Training Start", verbose=False)
        
        # Create unique seed for this client and round to ensure reproducibility
        client_seed = base_seed + self.client_id * 1000 + round_idx

        # Setup DataLoader with deterministic generator
        local_generator = torch.Generator().manual_seed(client_seed)
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=local_generator
        )
      
        has_active_attack = self._has_active_attack_this_round(round_idx)
        
        if has_active_attack:
            epochs = self.config.get('adversarial_epochs', 6)
            for attack in self.attacks: # should only be one IBA attack
            # Special handling for generative attacks that need pre-training
                if hasattr(attack, 'train_attack_model'):
                    attack.train_attack_model(self.model, dataloader, self.client_id, self.device, verbose=True)
        
       
        # Training loop
        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        start_time = time.time() # remove later
        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                # error when use resnet when batch size is 1; as data: torch.Size([1, 3, 32, 32]), target: torch.Size([1])
                # Error during experiment: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])
                # skip this batch
                if data.size(0) == 1:
                    continue
            
                if has_active_attack:
                    for attack in self.attacks:
                        apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
                        if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
                            data, target = attack.poison_data(data, target)
                    
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item() * data.size(0)
                epoch_correct += (output.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)
                
                # Memory cleanup
                if batch_idx % 200 == 0:
                    MemoryMonitor.cleanup_memory()
            
            # Epoch statistics
            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples

            if epoch == epochs - 1:
                print(f"   Client {self.client_id}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Time: {time.time() - start_time:.2f}s")
        

        # print l2 norm of the model
        # model_norm = torch.norm(torch.cat([p.view(-1) for p in self.model.parameters()]), p=2).item()
        # print(f"Client {self.client_id}: model_norm after training: {model_norm}")
        # print(f"*"*30)

        # with torch.no_grad():
        #     model_cpu = self.model.cpu()
        #     model_state = model_cpu.state_dict()
        #     del model_cpu
        #
        #     # === 新增：在此处拦截，执行模型投毒（梯度掩码、缩放等） ===
        #     if has_active_attack:
        #         for attack in self.attacks:
        #             apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
        #             if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):
        #                 # 获取当前的聚合算法名称 (如 FedAvg)
        #                 agg_algo = self.config.get('server_aggregation', [{'name': 'FedAvg'}])[0]['name']
        #
        #                 # 调用攻击类中的模型投毒方法，篡改最终要上传的 model_state
        #                 model_state = attack.apply_model_poisoning(
        #                     local_model_state=model_state,
        #                     global_model_state=self.global_model_state,
        #                     algorithm=agg_algo
        #                 )
        with torch.no_grad():
            model_cpu = self.model.cpu()
            model_state = model_cpu.state_dict()
            del model_cpu

            # === 新增：在此处拦截，执行模型投毒（梯度掩码、缩放等） ===
            if has_active_attack:
                for attack in self.attacks:
                    apply_to_client_ids = attack.config.get('apply_to_client_ids', [])
                    if attack and self.client_id in apply_to_client_ids and attack.should_apply(round_idx):

                        # =======================================================
                        # 💥 【第二步修改：为 LP 攻击动态挂载 LSA 测试环境】
                        # =======================================================
                        if hasattr(attack, 'setup_lsa_environment'):
                            # 注意：因为上面刚执行了 self.model.cpu()，模型现在在 CPU 上
                            # 所以我们将推理设备指定为 torch.device('cpu')，
                            # 测试一两个 Batch 非常快，还能避免显存报错。
                            attack.setup_lsa_environment(
                                model=self.model,
                                dataloader=dataloader,
                                device=self.device
                            )
                        # =======================================================

                        # 获取当前的聚合算法名称 (如 FedAvg)
                        agg_algo = self.config.get('server_aggregation', [{'name': 'FedAvg'}])[0]['name']

                        # 调用攻击类中的模型投毒方法，篡改最终要上传的 model_state
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
            # 'attacks_applied': [attack.name for attack in self.attacks if attack and self.client_id in attack.config.get('apply_to_client_ids', []) and attack.should_apply(round_idx)],
            # 'defenses_applied': [defense.name for defense in self.defenses if defense.should_apply(round_idx)],
        }
        
        # Cleanup
        self.cleanup_memory()
        
        return result
    
    def cleanup_memory(self):
        """Clean up client memory"""
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None

         # === 新增：清理全局副本 ===
        if hasattr(self, 'global_model_state'):
            del self.global_model_state


        
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()