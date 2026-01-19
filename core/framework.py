"""
Main FL Research Framework
"""

import torch
import numpy as np
import gc
import os
from typing import Dict, Any
from .memory import MemoryMonitor
from .client import FLClient
from .server import FLServer
from .models import create_model_from_dataset_config
import time

class FLResearchFramework:
    """Main framework class that orchestrates everything"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = f"cuda:{config['gpu']['id']}" if torch.cuda.is_available() else 'cpu'
        
        # Initialize memory monitoring
        MemoryMonitor.set_gpu_id(config['gpu']['id'])
        MemoryMonitor.init_cpu_baseline()
        
        # Initialize components
        self.global_model = None
        self.clients = []
        self.server = None
        
        print(f"🚀 FL Research Framework initialized")
        print(f"   Device: {self.device}")
        print(f"   Dataset: {config['dataset']['name']}")
        print(f"   Model: {config['model']['name']}")
    
    def setup_experiment(self, train_loader, test_loader, client_datasets, checkpoint_path: str = None):
        """Setup the FL experiment with optional checkpoint loading"""
        print(f"🏗️ Setting up experiment...")
        
        # Create global model
        self.global_model = create_model_from_dataset_config(self.config)
        
        # Create server
        self.server = FLServer(self.global_model, self.device, self.config)
        
        round_num = 0
        checkpoint_data = None
        
        # Load checkpoint if provided and exists
        if checkpoint_path and checkpoint_path.strip() and os.path.exists(checkpoint_path):
            try:
                checkpoint_data = self.server.load_checkpoint(checkpoint_path)
                round_num = checkpoint_data.get('round_num', 0)
                print(f"✅ Checkpoint loaded: Round {round_num}")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                print(f"🔄 Starting training from scratch (round 0)")
                checkpoint_data = None
                round_num = 0
        elif checkpoint_path and checkpoint_path.strip():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            print(f"🔄 Starting training from scratch (round 0)")
            round_num = 0
        else:
            print(f"🔄 No checkpoint specified, starting training from scratch (round 0)")
            round_num = 0
        
        # Create clients
        print(f"Number of clients: {len(client_datasets)}")
        self.clients = []
        for i, client_dataset in enumerate(client_datasets):
            client = FLClient(i, client_dataset, self.device, self.config)
            
            # Load attack models if checkpoint data is available
            if checkpoint_data and 'attack_models' in checkpoint_data:
                client.load_attack_models(checkpoint_data['attack_models'])
            
            self.clients.append(client)
        
        print(f"✅ Experiment setup complete!")
        print(f"=="*30)
        return round_num
    
    def run_training_round(self, selected_clients, round_idx):
        """Run one training round"""
        
        # Reset peak memory tracking
        MemoryMonitor.reset_peaks()
        
        # Distribute global model to clients with timing
        distribute_start_time = time.time()
        # Distribute global model to clients
        for client in selected_clients:
            client.set_model(self.global_model)
        distribute_time = time.time() - distribute_start_time
        # print client ids participating in the round
        print(f"Client ids participating in the round: {[client.client_id for client in selected_clients]}")

        # Local training with timing
        client_results = []
        client_training_times = []
        
        for i, client in enumerate(selected_clients):
            client_start_time = time.time()
            # MemoryMonitor.monitor_memory("Client Number " + str(i) + " Start")
            result = client.train(
                epochs=self.config['federated_learning']['local_epochs'],
                batch_size=self.config['federated_learning']['batch_size'],
                round_idx=round_idx,
                base_seed=self.config['experiment']['seed']
            )
            client_training_time = time.time() - client_start_time
            client_training_times.append(client_training_time)
            client_results.append(result)
            # MemoryMonitor.monitor_memory("Client Number " + str(i) + " Train End")
            MemoryMonitor.cleanup_memory(aggressive=True)
        
        # Model aggregation with timing
        aggregation_start_time = time.time()
        self.global_model = self.server.aggregate_models(client_results, round_idx)
        aggregation_time = time.time() - aggregation_start_time
        
        # Calculate round statistics
        avg_accuracy = np.mean([r['accuracy'] for r in client_results])
        avg_loss = np.mean([r['loss'] for r in client_results])
        total_samples = sum([r['samples'] for r in client_results])
        
        # Get peak memory
        peak_cpu, peak_gpu = MemoryMonitor.get_round_peaks()
        
        # Calculate timing metrics
        total_round_time = distribute_time + sum(client_training_times) + aggregation_time
        max_client_training_time = max(client_training_times) if client_training_times else 0
        minimal_time = distribute_time + max_client_training_time + aggregation_time
        
        round_metrics = {
            'round': round_idx + 1,
            'train_accuracy': float(avg_accuracy),
            'train_loss': float(avg_loss),
            'total_samples': int(total_samples),
            'selected_clients': [c.client_id for c in selected_clients],
            'benign_clients': [r['client_id'] for r in client_results if not r['active_attack']],
            'adversarial_clients': [r['client_id'] for r in client_results if r['active_attack']],
            'peak_cpu_memory_gb': float(peak_cpu),
            'peak_gpu_memory_gb': float(peak_gpu),
            'total_round_time_seconds': float(total_round_time),
            'minimal_time_seconds': float(minimal_time),
            'distribute_time_seconds': float(distribute_time),
            'client_training_times': client_training_times,
            'aggregation_time_seconds': float(aggregation_time),
        }

        print(f"Round {round_idx + 1} benign clients: {round_metrics['benign_clients']} and adversarial clients: {round_metrics['adversarial_clients']} and total clients: {len(selected_clients)}")
        
        # Memory cleanup
        MemoryMonitor.cleanup_memory(aggressive=True)
        gc.collect()
        
        return round_metrics

    def evaluate(self, test_loader):
        """Evaluate the global model"""
        return self.server.evaluate(test_loader)
    
    def evaluate_backdoor(self, test_loader, attack_config):
        """Evaluate the global model with backdoor triggers"""
        # print(f"🔍 Evaluating backdoor: {attack_config}")
        return self.server.evaluate_backdoor(test_loader, attack_config)
    
    def dump_backdoor_visualization(self, test_loader, attack_config):
        """Dump backdoor visualization"""
        return self.server.dump_backdoor_visualization(test_loader, attack_config)
        
    def cleanup_memory(self):
        """Clean up all memory"""
        for client in self.clients:
            client.cleanup_memory()
        
        if self.server:
            self.server.cleanup_memory()
        
        MemoryMonitor.cleanup_memory(aggressive=True)
        gc.collect()
