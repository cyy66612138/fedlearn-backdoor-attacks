"""
FL Server implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple
from datetime import datetime
from .memory import MemoryMonitor
from .aggregations import create_aggregation_method
from .defenses import create_defense
from .models import create_model_from_dataset_config


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
        
        # TODO: check only first aggregation method is used, select aggregation method for this round
        agg_method = self.aggregation_methods[0]  # Use first method for now
        
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
        
        # Move to CPU to save memory
        aggregated_model = aggregated_model.cpu()
        
        return aggregated_model
    
    def _results_to_models(self, client_results: List[Dict]) -> List[nn.Module]:
        """Convert client results to model instances"""
        models = []
        for result in client_results:
            # Create model instance
            model = create_model_from_dataset_config(self.config)
            
            # Load state dict
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
        
        # Print first 4 elements of the first layer of the global model
        # first_layer = list(self.global_model.parameters())[0]
        # print("First 4 elements of the first layer:", first_layer.view(-1)[:4].tolist())
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                    
                # if total_samples == 0:
                #     print(f"🔍 Server: Evaluating data shape: {data.shape} and target shape: {target.shape}")
                #     print(f"Max data: {data.max()}, Min data: {data.min()}, Max target: {target.max()}, Min target: {target.min()}")
                    
                output = self.global_model(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item() * data.size(0)
                correct += (output.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

                
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        # Move back to CPU
        self.global_model = self.global_model.cpu()
        
        MemoryMonitor.monitor_memory("After Evaluation", verbose=False)
        gc.collect()
        return accuracy, avg_loss, total_samples
    
    # def _clone_config(self, config: dict) -> dict:
    #     # Shallow copy is sufficient given the simple edits we perform
    #     return {k: (v.copy() if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in config.items()}
        
    def _add_attack_config_norm(self, attack_config: Dict) -> Dict:
        """Add normalization values to attack config"""
        dataset_config = self.config['dataset']
        attack_config_with_norm = attack_config.copy()
        attack_config_with_norm['dataset_name'] = dataset_config['name']
        attack_config_with_norm['num_classes'] = dataset_config['num_classes']
        attack_config_with_norm['mean'] = dataset_config['mean']
        attack_config_with_norm['std'] = dataset_config['std']
        attack_config_with_norm['client_id'] = -1 # -1 means evaluation, so poison all data in each batch
        attack_config_with_norm['input_dim'] = dataset_config['data_shape'][1] # input dim is the second element of data_shape
        attack_config_with_norm['attack_start_round'] = self.config['federated_learning']['attack_start_round']
        attack_config_with_norm['attack_stop_round'] = self.config['federated_learning']['attack_stop_round']
        attack_config_with_norm['attack_frequency'] = self.config['federated_learning']['attack_frequency']
        attack_config_with_norm['seed'] = self.config.get('experiment', {}).get('seed', 42)
        return attack_config_with_norm
    
    def evaluate_backdoor(self, test_loader: DataLoader, attack_config: Dict) -> Tuple[float, float]:
        """Evaluate global model with backdoor triggers"""
        from .attacks import create_attack
        
        MemoryMonitor.monitor_memory("Before Backdoor Evaluation", verbose=False)

        # print l2 norm of the global model
        global_model_params = list(self.global_model.parameters())
        global_model_norm = torch.norm(torch.cat([p.view(-1) for p in global_model_params]), p=2)
        print(f"🔍 Server: evaluate_backdoor global_model_norm: {global_model_norm}")
        
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
        
        attack_config_with_norm = self._add_attack_config_norm(attack_config)
        # print(f"🔍 Server: attack_config_with_norm: {attack_config_with_norm}")
        # Create attack for backdoor testing
        attack = create_attack(attack_config_with_norm)
        
        # Check if attack uses fixed target class or random target class
        uses_random_target_label =  attack_config_with_norm.get('target_class', -1) == -1
        print(f"🔍 Server: evaluate_backdoor name: {attack_config_with_norm.get('name', '')} with uses_random_target_label: {uses_random_target_label}")
        total_loss, correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                if uses_random_target_label:
                    # For random target class attacks, use all samples
                    data_filtered = data
                    labels_filtered = target
                else:
                    # For fixed target class attacks, filter out samples that already have target class
                    target_class = attack_config_with_norm.get('target_class', 0)
                    non_target_mask = (target != target_class)
                    if non_target_mask.sum() == 0:
                        continue  # Skip batch if all samples have target class
                    # Keep only non-target samples for evaluation
                    data_filtered = data[non_target_mask]
                    labels_filtered = target[non_target_mask]
                if len(data_filtered) == 0:
                    continue
                # print(f"Len data_filtered: {data_filtered.size(0)}, len labels_filtered: {labels_filtered.size(0)}")
                # Apply backdoor trigger to test data
                poisoned_data, poisoned_labels = attack.poison_data(data_filtered, labels_filtered)

                output = self.global_model(poisoned_data)
                loss = F.cross_entropy(output, poisoned_labels)
                
                # Count correct predictions (should predict poisoned_labels after poisoning)
                predicted = output.argmax(dim=1)
                correct_predictions = (predicted == poisoned_labels).sum().item()
                
                total_loss += loss.item() * data_filtered.size(0)
                correct += correct_predictions
                total_samples += data_filtered.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        # Move back to CPU
        self.global_model = self.global_model.cpu()
        # check if using atk_model then move it to cpu
        if attack.use_model_trigger():
            attack.atk_model = attack.atk_model.cpu()
        
        MemoryMonitor.monitor_memory("After Backdoor Evaluation", verbose=False)
        gc.collect()
        
        return accuracy, avg_loss, total_samples

    
    def dump_backdoor_visualization(self, test_loader: DataLoader, attack_config: Dict) -> None:
        """Dump backdoor visualization"""
        import pickle
        from .attacks import create_attack
        
        MemoryMonitor.monitor_memory("Before Backdoor Dump Visualization", verbose=False)
        
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
       
        attack_config_with_norm = self._add_attack_config_norm(attack_config)
        # Prepare dump directory
        visualize_dir = self.config['logging']['visualize_dir']
        attack_config_with_norm['dump_path'] = visualize_dir

        # Check if attack uses fixed target class or random target class
        uses_random_target_label =  attack_config_with_norm.get('target_class', -1) == -1
        print(f"🔍 Server: evaluate_backdoor name: {attack_config_with_norm.get('name', '')} with uses_random_target_label: {uses_random_target_label}")
        total_loss, correct, total_samples = 0.0, 0, 0
        
        data_pickle = {}
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Only dump for the first batch
                if batch_idx > 0:
                    break

                data, target = data.to(self.device), target.to(self.device)
                
                if uses_random_target_label:
                    # For random target class attacks, use all samples
                    data_filtered = data
                    labels_filtered = target
                    # then genrate for all target labels
                    for target_label in range(attack_config_with_norm['num_classes']):
                        print(f"🔍 Server (Random Target Class) dumping backdoor visualization: {target_label}")
                        attack_config_with_norm['target_class'] = target_label
                        attack = create_attack(attack_config_with_norm)
                        poisoned_data, poisoned_labels = attack.poison_data(data_filtered, labels_filtered)
                        
                        clean_imgs = attack._denormalize(data_filtered)
                        poisoned_imgs = attack._denormalize(poisoned_data)
                        noise_imgs = poisoned_imgs - clean_imgs
                        
                        data_pickle[f'labels_{target_label}'] = {
                            'clean_imgs': clean_imgs.cpu().numpy(),
                            'poisoned_imgs': poisoned_imgs.cpu().numpy(),
                            'noise_imgs': noise_imgs.cpu().numpy(),
                            'clean_labels': labels_filtered.cpu().numpy(),
                            'poisoned_labels': poisoned_labels.cpu().numpy()
                        }
                    # duplicate the first image into batch size
                    first_img = data_filtered[0:1].clone()  # Keep shape (1, C, H, W)
                    data_duplicated = first_img.repeat(len(data_filtered), 1, 1, 1)  # (batch_size, C, H, W)
                    labels_duplicated = labels_filtered[0:1].repeat(len(labels_filtered))  # (batch_size,)
                    
                    # Generate poison for all target labels using duplicated first image
                    for target_label in range(attack_config_with_norm['num_classes']):
                        print(f"🔍 Server (Random Target Class) dumping backdoor visualization (duplicated): {target_label}")
                        attack_config_with_norm['target_class'] = target_label
                        attack = create_attack(attack_config_with_norm)
                        
                        # Poison the duplicated data
                        poisoned_data_dup, poisoned_labels_dup = attack.poison_data(data_duplicated, labels_duplicated)
                        
                        clean_imgs_dup = attack._denormalize(data_duplicated)
                        poisoned_imgs_dup = attack._denormalize(poisoned_data_dup)
                        noise_imgs_dup = poisoned_imgs_dup - clean_imgs_dup
                        
                        data_pickle[f'labels_{target_label}_dup'] = {
                            'clean_imgs': clean_imgs_dup.cpu().numpy(),
                            'poisoned_imgs': poisoned_imgs_dup.cpu().numpy(),
                            'noise_imgs': noise_imgs_dup.cpu().numpy(),
                            'clean_labels': labels_duplicated.cpu().numpy(),
                            'poisoned_labels': poisoned_labels_dup.cpu().numpy()
                        }
                    
                    target_label = -1
                    attack_config_with_norm['target_class'] = target_label
                    attack = create_attack(attack_config_with_norm)
                    poisoned_data, poisoned_labels = attack.poison_data(data_filtered, labels_filtered)

                    
                    
                else:
                    data_filtered = data
                    labels_filtered = target
                    # For fixed target class attacks, filter out samples that already have target class
                    target_class = attack_config_with_norm['target_class']
                    # Apply backdoor trigger to test data
                    print(f"🔍 Server (Fixed Target Class) dumping backdoor visualization: {target_class}")
                    attack = create_attack(attack_config_with_norm)
                    poisoned_data, poisoned_labels = attack.poison_data(data_filtered, labels_filtered)

                    # Denormalize to [0,1] for visualization using existing attack helpers
                    clean_imgs = attack._denormalize(data_filtered)
                    poisoned_imgs = attack._denormalize(poisoned_data)
                    noise_imgs = poisoned_imgs - clean_imgs
                    
                    data_pickle[f'labels_{target_class}'] = {
                        'clean_imgs': clean_imgs.cpu().numpy(),
                        'poisoned_imgs': poisoned_imgs.cpu().numpy(),
                        'noise_imgs': noise_imgs.cpu().numpy(),
                        'clean_labels': labels_filtered.cpu().numpy(),
                        'poisoned_labels': poisoned_labels.cpu().numpy()
                    }
                    # from .attacks import dump_poisoned_images
                    # dump_poisoned_images(clean_imgs=clean_imgs, clean_labels=labels_filtered, poison_indices=None, poisoned_imgs=poisoned_imgs, poisoned_labels=poisoned_labels, noise_imgs=noise_imgs, attack_config=attack_config_with_norm)
                
                # evaluate accuracy and loss
                output = self.global_model(poisoned_data)
                loss = F.cross_entropy(output, poisoned_labels)
                predicted = output.argmax(dim=1)
                correct_predictions = (predicted == poisoned_labels).sum().item()
                total_loss += loss.item() * data_filtered.size(0)
                correct += correct_predictions
                total_samples += data_filtered.size(0)

        dataset_name = attack_config_with_norm['dataset_name']
        attack_name = attack_config_with_norm['name']
        dump_path = attack_config_with_norm['dump_path']
        target_class = attack_config_with_norm['target_class']
        eps = attack_config_with_norm.get('atk_eps', -1)
        latent_dim = attack_config_with_norm.get('atk_latent_dim', -1)
        fn = f'{dataset_name}_{attack_name}_labels_{target_class}_eps_{eps}_latent_dim_{latent_dim}_images.pickle'
        os.makedirs(dump_path, exist_ok=True)
        # save data_pickle to pickle file   
        with open(os.path.join(dump_path, fn), 'wb') as f:
            pickle.dump(data_pickle, f)
        print(f"🖼️ Saved data_pickle to {os.path.join(dump_path, fn)}")
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        # Move back to CPU
        self.global_model = self.global_model.cpu()
        # check if using atk_model then move it to cpu
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
            
            # Save model state dict on CPU (best practice)
            experiment_name = self.config.get('experiment', {}).get('name', 'default_experiment')
            checkpoint_path = f"{self.checkpoint_dir}/{experiment_name}__round_{round_num}.pth"
            
            # Move model to CPU for saving, then back to original device
            model_cpu = self.global_model.cpu()
            
            checkpoint_data = {
                'round_num': round_num,
                'model_state_dict': model_cpu.state_dict(),  # Save on CPU
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
            }
            
            # Collect attack models from class caches if not provided
            if attack_models is None:
                attack_models = self._collect_attack_models_from_caches()
            
            # Save attack models if available
            if attack_models:
                # print(f"🔍 Saving {len(attack_models)} attack models with skeys {list(attack_models.keys())} in checkpoint")
                attack_models_cpu = {}
                for attack_name, attack_data in attack_models.items():
                    model = attack_data.get('model')
                    if model is not None:
                        # Move model to CPU for saving
                        model_cpu = model.cpu()
                        attack_models_cpu[attack_name] = {
                            'model_state_dict': model_cpu.state_dict(),
                            'attack_config': attack_data['config'],
                            'attack_class': attack_data['attack_class']
                        }
                        
                if attack_models_cpu:
                    checkpoint_data['attack_models'] = attack_models_cpu
                    print(f"💾 Saving {len(attack_models_cpu)} attack models with keys {list(attack_models_cpu.keys())} in checkpoint")
                
                # Clear attack_models to free memory
                del attack_models
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Clear checkpoint data to free memory
            del checkpoint_data
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"💾 Checkpoint saved: {checkpoint_path} at round {round_num} and timestamp: {datetime.now().isoformat()}")
            return checkpoint_path
        
        return None
    
    def _collect_attack_models_from_caches(self) -> Dict[str, Any]:
        """Collect attack models from class-level caches"""
        attack_models = {}
        
        try:
            # Get attack configs from experiment
            attack_configs = self.config['client_attacks']
            dataset_name = self.config['dataset']['name'] # get dataset name from config

            for id_atk, atk_config in enumerate(attack_configs):
                attack_name = atk_config.get('name', '')    
                
                # Note: IBAAttack, MarksmanAttack, and PolyMorphAttack are no longer supported
                # Add support for other attacks here if they have model caches
                
                if attack_name not in ['IBAAttack', 'MarksmanAttack', 'PolyMorphAttack']:
                    print(f"⚠️ Unsupported attack or attack without model cache: {attack_name}")
            
            if attack_models:
                print(f"🔍 In _collect_attack_models_from_caches: Found {len(attack_models)} cached attack models")
            else:
                print("ℹ️ In _collect_attack_models_from_caches: No cached attack models found")
        except Exception as e:
            print(f"❌ Failed to collect attack models from caches: {str(e)}")
            return {}
        
        return attack_models
            
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint including generative models for attacks that require generators"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint (model state dict is on CPU)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state dict (checkpoint was saved on CPU)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load attack models if they exist in checkpoint
        if 'attack_models' in checkpoint:
            print(f"📂 Found {len(checkpoint['attack_models'])} attack models in checkpoint with keys {list(checkpoint['attack_models'].keys())}")
            # Return attack models data for external use
            # checkpoint['loaded_attack_models'] = checkpoint['attack_models']
        else:
            print("ℹ️ No attack models found in checkpoint")

        print(f"📂 Checkpoint loaded: {checkpoint_path} Round: {checkpoint['round_num']} and timestamp: {checkpoint['timestamp']}")
        
        return checkpoint
        
    def cleanup_memory(self):
        """Clean up server memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()