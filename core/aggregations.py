"""
Aggregation methods for federated learning
"""
# ==============================================================================
# 🟢 【终极拦截】必须放在文件第 1 行！在加载任何其他库之前拦截 scikit-learn
# ==============================================================================
import sklearn.utils.validation
import sklearn.utils

_original_check_array = sklearn.utils.validation.check_array

def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _original_check_array(*args, **kwargs)

# 暴力覆盖 sklearn 内部所有的 check_array 引用点
sklearn.utils.validation.check_array = _patched_check_array
sklearn.utils.check_array = _patched_check_array
# ==============================================================================

# 下面才是你原本的 imports
# import numpy as np
# import torch
# import hdbscan
# ... (其余代码保持不变)
import torch
import torch.nn as nn
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
import copy
import math
import random
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.cluster import estimate_bandwidth

# ==============================================================================
# 🟢 【Hotfix】解决 scikit-learn 1.3+ 版本废弃 force_all_finite 导致的 FLAME/HDBSCAN 报错
# ==============================================================================
import sklearn.utils.validation as validation
import sklearn.metrics.pairwise as pairwise

# 1. 拦截并修复 check_array
original_check_array = validation.check_array
def patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        # 将被废弃的 force_all_finite 替换为新版的 ensure_all_finite
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return original_check_array(*args, **kwargs)
validation.check_array = patched_check_array

# 2. 拦截并修复 cosine_distances (如果在 FLAME 中直接调用了)
original_cosine_distances = pairwise.cosine_distances
def patched_cosine_distances(X, Y=None, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs.pop('force_all_finite')  # 新版直接移除了该参数，安全丢弃
    return original_cosine_distances(X, Y, **kwargs)
pairwise.cosine_distances = patched_cosine_distances
# ==============================================================================

def compute_l2_distance(state_dict1, state_dict2):
    # Create a list of flattened tensors of the difference between each layer
    diffs = [ (state_dict1[key].float() - state_dict2[key].float()).view(-1) for key in state_dict1 ]
    # Concatenate all the tensors into a single flat tensor
    flat_diff = torch.cat(diffs)
    # Compute the L2 norm (Euclidean norm)
    return torch.norm(flat_diff, p=2).item()

class BaseAggregation(ABC):
    """Base class for aggregation methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
    
    @abstractmethod
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        pass

    def print_first_layer(self, global_model: nn.Module, top_k: int = 4):
        first_layer = list(global_model.parameters())[0]
        print(f"First {top_k} elements of the first layer of the global model: {first_layer.view(-1)[:top_k].tolist()}")

    def print_l2_distance(self, global_model: nn.Module, client_results: List[Dict]):
        l2_distance = [compute_l2_distance(result['model_state'], global_model.state_dict()) for result in client_results ]
        print(f"L2 distance between each client and global model: {l2_distance}")

    def print_weight_of_each_client(self, client_results: List[Dict]):
        total_samples = sum(result['samples'] for result in client_results)
        weights = [(result['samples'] / total_samples) for result in client_results]
        print(f"Weight of each client: {[f'{w:.4f}' for w in weights]}")

class FedAvgAggregation(BaseAggregation):
    """FedAvg aggregation"""
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
              round_num: int, verbose: bool = False) -> nn.Module:
        """
        Aggregates client updates using:
        global_model += sum_i (client_ratio  * (local_model_i - global_model))
        """
        if not client_results:
            return global_model
        
        total_samples = sum(result['samples'] for result in client_results)
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()

        if verbose:
            print("Before aggregation:")
            self.print_first_layer(global_model, top_k=4)
            self.print_l2_distance(global_model, client_results)
            self.print_weight_of_each_client(client_results)        

        # Create a detached copy of global model to compute deltas from
        global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Prepare update accumulator
        aggregated_update = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_snapshot.items()}
        
        
        # Accumulate updates from each client
        with torch.no_grad():
            for result in client_results:
                weight = result['samples'] / total_samples
                local_state = result['model_state']
                
                for key in model_keys:
                    local_tensor = local_state[key].float()
                    delta = local_tensor - global_snapshot[key]
                    aggregated_update[key] += weight * delta

            # Apply updates to the actual global model
            for key in model_keys:
                # Ensure dtype compatibility before adding
                if global_state[key].dtype != aggregated_update[key].dtype:
                    # Convert aggregated_update to match global_state dtype
                    aggregated_update[key] = aggregated_update[key].to(global_state[key].dtype)
                global_state[key].add_(aggregated_update[key])


        global_model.load_state_dict(global_state)

        if verbose:
            print("After aggregation:")
            self.print_first_layer(global_model, top_k=4)

        return global_model

class FedSGDAggregation(BaseAggregation):
    """
    FedSGD (Federated Stochastic Gradient Descent) aggregation
    
    [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a) - AISTATS '17
    
    FedSGD aggregates client gradients (updates) instead of full model parameters.
    Clients perform single local epoch and send gradients (local_model - global_model).
    Server aggregates gradients and adds to global model.
    
    This follows the same computation as FedAvg but conceptually represents
    gradient-based aggregation (used by FedSGD algorithm).
    """
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
              round_num: int, verbose: bool = False) -> nn.Module:
        """
        Aggregates client gradients (updates) using:
        global_model += sum_i (client_ratio * (local_model_i - global_model))
        
        Args:
            global_model: The global model to update
            client_results: List of client results, each containing:
                - 'model_state': Dict[str, torch.Tensor] - client's model state dict
                - 'samples': int - number of training samples
            round_num: Current round number
            verbose: Whether to print debug information
        
        Returns:
            Updated global model
        """
        if not client_results:
            return global_model
        
        total_samples = sum(result['samples'] for result in client_results)
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()

        if verbose:
            print("Before aggregation:")
            self.print_first_layer(global_model, top_k=4)
            self.print_l2_distance(global_model, client_results)
            self.print_weight_of_each_client(client_results)        

        # Create a detached copy of global model to compute gradients (updates) from
        global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Prepare gradient accumulator
        aggregated_gradient = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_snapshot.items()}
        
        # Accumulate gradients (updates) from each client
        with torch.no_grad():
            for result in client_results:
                weight = result['samples'] / total_samples
                local_state = result['model_state']
                
                for key in model_keys:
                    local_tensor = local_state[key].float()
                    # Compute gradient (update): local - global
                    gradient = local_tensor - global_snapshot[key]
                    aggregated_gradient[key] += weight * gradient

            # Apply aggregated gradients to the global model
            for key in model_keys:
                # Ensure dtype compatibility before adding
                if global_state[key].dtype != aggregated_gradient[key].dtype:
                    # Convert aggregated_gradient to match global_state dtype
                    aggregated_gradient[key] = aggregated_gradient[key].to(global_state[key].dtype)
                # Add aggregated gradient to global model: global += aggregated_gradient
                global_state[key].add_(aggregated_gradient[key])

        global_model.load_state_dict(global_state)

        if verbose:
            print("After aggregation:")
            self.print_first_layer(global_model, top_k=4)

        return global_model

class FedProxAggregation(BaseAggregation):
    """
    FedProx aggregation
    
    [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) - MLSys '20
    
    FedProx uses the same aggregation as FedAvg. The key difference is on the client side,
    where a proximal term is added to the loss to prevent clients from deviating too far
    from the global model. The server-side aggregation is identical to FedAvg.
    
    Note: The proximal term (mu * ||w - w_t||^2) is applied during client training,
    not during aggregation. This aggregation method performs standard FedAvg aggregation.
    """
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
              round_num: int, verbose: bool = False) -> nn.Module:
        """
        Aggregates client updates using FedAvg (same as FedProx server-side).
        
        Args:
            global_model: The global model to update
            client_results: List of client results, each containing:
                - 'model_state': Dict[str, torch.Tensor] - client's model state dict
                - 'samples': int - number of training samples
            round_num: Current round number
            verbose: Whether to print debug information
        
        Returns:
            Updated global model
        """
        # FedProx uses the same aggregation as FedAvg
        fedavg = FedAvgAggregation(self.config)
        return fedavg.aggregate(global_model, client_results, round_num, verbose)


class SCAFFOLDAggregation(BaseAggregation):
    """
    SCAFFOLD aggregation
    
    [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) - ICML '20
    
    SCAFFOLD uses control variates to correct for client drift. The server maintains
    global control variates (c_global) and aggregates client updates with a global learning rate.
    
    Steps per round:
      1) Aggregate client model updates (y_delta) with global_lr
      2) Update global control variates (c_global) based on client control variate differences (c_delta)
    
    Config keys (under params):
      - global_lr: float (default 1.0) - server learning rate for aggregating updates
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.global_lr = float(self.params.get('global_lr', 1.0))
        # Persistent state: global control variates (one per model parameter)
        self.c_global: List[torch.Tensor] = []
        print(f"🔍 SCAFFOLDAggregation: global_lr: {self.global_lr}")
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int, verbose: bool = False) -> nn.Module:
        """
        Aggregates client updates using SCAFFOLD method.
        
        Args:
            global_model: The global model to update
            client_results: List of client results, each containing:
                - 'model_state': Dict[str, torch.Tensor] - client's model state dict (or 'y_delta' if return_diff=True)
                - 'samples': int - number of training samples
                - 'y_delta': List[torch.Tensor] - model parameter differences (required for SCAFFOLD)
                - 'c_delta': List[torch.Tensor] - control variate differences (required for SCAFFOLD)
            round_num: Current round number
            verbose: Whether to print debug information
        
        Returns:
            Updated global model
        """
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        
        # Initialize c_global if first round
        if len(self.c_global) == 0:
            self.c_global = [
                torch.zeros_like(param, dtype=torch.float32) 
                for param in global_state.values()
            ]
        
        # Extract y_delta and c_delta from client results
        # SCAFFOLD expects clients to return differences, not full model states
        y_delta_list = []
        c_delta_list = []
        weights = []
        
        total_samples = sum(result['samples'] for result in client_results)
        
        for result in client_results:
            weight = result['samples'] / total_samples
            weights.append(weight)
            
            # Get y_delta (model parameter differences)
            if 'y_delta' in result:
                y_delta_list.append(result['y_delta'])
            else:
                # Fallback: compute delta from model_state if y_delta not provided
                # This handles cases where clients don't explicitly return differences
                global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
                y_delta = []
                for key in model_keys:
                    local_tensor = result['model_state'][key].float()
                    delta = local_tensor - global_snapshot[key]
                    y_delta.append(delta)
                y_delta_list.append(y_delta)
            
            # Get c_delta (control variate differences)
            if 'c_delta' in result:
                c_delta_list.append(result['c_delta'])
            else:
                # If c_delta not provided, use zeros (fallback for compatibility)
                c_delta_list.append([
                    torch.zeros_like(c_g) for c_g in self.c_global
                ])
        
        # Convert weights to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        # Aggregate model updates with global_lr
        # Following FL-bench: param.data += global_lr * sum(weights * y_delta)
        with torch.no_grad():
            # Transpose y_delta_list: from [client][param] to [param][client]
            y_delta_by_param = list(zip(*y_delta_list))
            
            for param_idx, (key, global_param) in enumerate(zip(model_keys, global_state.values())):
                # Get y_deltas for this parameter from all clients
                y_deltas_for_param = list(y_delta_by_param[param_idx])
                
                # Stack along last dimension: shape becomes [..., num_clients]
                # Following FL-bench: torch.stack(y_delta, dim=-1) * weights
                y_deltas_stacked = torch.stack(y_deltas_for_param, dim=-1)
                
                # Weighted sum: sum(weights * y_deltas) along last dimension
                # weights has shape [num_clients], broadcasts correctly
                aggregated_delta = torch.sum(y_deltas_stacked * weights_tensor, dim=-1)
                
                # Apply with global_lr: param += global_lr * aggregated_delta
                update = (self.global_lr * aggregated_delta).to(global_param.dtype)
                global_param.add_(update)
            
            # Update global control variates
            # Following FL-bench: c_global += sum(c_delta) / num_clients
            num_clients = len(c_delta_list)
            for c_global, c_deltas in zip(self.c_global, zip(*c_delta_list)):
                # Stack c_deltas: [num_clients, ...]
                c_deltas_stacked = torch.stack(c_deltas, dim=-1)
                
                # Average: sum(c_delta) / num_clients
                c_global_update = c_deltas_stacked.sum(dim=-1) / num_clients
                c_global.add_(c_global_update.to(c_global.dtype))
        
        global_model.load_state_dict(global_state)
        return global_model


class FedOptAggregation(BaseAggregation):
    """
    FedOpt (Federated Optimization) aggregation
    
    [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295) - arxiv '20, ICLR '21
    
    FedOpt aggregates client gradients (updates) with adaptive optimization on the server.
    The server uses adaptive optimizers (Adagrad, Yogi, or Adam) to update the global model.
    
    Steps per round:
      1) Compute weighted average of client model differences (deltas)
      2) Update momentums using beta1
      3) Update velocities using the selected adaptive optimizer rule (Adagrad/Yogi/Adam)
      4) Update global model: param += lr * (momentum / (sqrt(velocity) + tau))
    
    Config keys (under params):
      - type: str (default "adam") - optimizer type: "adagrad", "yogi", or "adam"
      - beta1: float (default 0.9) - momentum decay factor
      - beta2: float (default 0.999) - velocity decay factor (for Yogi/Adam)
      - server_lr: float (default 0.1) - server learning rate
      - tau: float (default 1e-3) - small constant for numerical stability
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.optimizer_type = str(self.params.get('type', 'adam')).lower()
        self.beta1 = float(self.params.get('beta1', 0.9))
        self.beta2 = float(self.params.get('beta2', 0.999))
        self.server_lr = float(self.params.get('server_lr', 0.1))
        self.tau = float(self.params.get('tau', 1e-3))
        
        if self.optimizer_type not in ['adagrad', 'yogi', 'adam']:
            raise ValueError(f"FedOpt optimizer type must be 'adagrad', 'yogi', or 'adam', got '{self.optimizer_type}'")
        
        # Persistent state: momentums and velocities (one per model parameter)
        self.momentums: List[torch.Tensor] = []
        self.velocities: List[torch.Tensor] = []
        
        print(f"🔍 FedOptAggregation: type={self.optimizer_type}, beta1={self.beta1}, "
              f"beta2={self.beta2}, server_lr={self.server_lr}, tau={self.tau}")
    
    def _update_velocity(self, velocity: torch.Tensor, delta: torch.Tensor):
        """Update velocity according to the selected optimizer type."""
        if self.optimizer_type == 'adagrad':
            # Adagrad: v = v + delta^2
            velocity.data.add_(delta ** 2)
        elif self.optimizer_type == 'yogi':
            # Yogi: v = v - (1 - beta2) * delta^2 * sign(v - delta^2)
            delta_pow2 = delta ** 2
            velocity.data.sub_((1 - self.beta2) * delta_pow2 * torch.sign(velocity - delta_pow2))
        elif self.optimizer_type == 'adam':
            # Adam: v = beta2 * v + (1 - beta2) * delta^2
            velocity.data.mul_(self.beta2).add_((1 - self.beta2) * delta ** 2)
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int, verbose: bool = False) -> nn.Module:
        """
        Aggregates client updates using FedOpt adaptive optimization.
        
        Args:
            global_model: The global model to update
            client_results: List of client results, each containing:
                - 'model_state': Dict[str, torch.Tensor] - client's model state dict
                  (or 'model_params_diff' if return_diff=True)
                - 'samples': int - number of training samples
            round_num: Current round number
            verbose: Whether to print debug information
        
        Returns:
            Updated global model
        """
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        
        # Initialize momentums and velocities if first round
        if len(self.momentums) == 0:
            self.momentums = [
                torch.zeros_like(param, dtype=torch.float32) 
                for param in global_state.values()
            ]
            self.velocities = [
                torch.zeros_like(param, dtype=torch.float32) 
                for param in global_state.values()
            ]
        
        # Compute weighted average of client model differences
        # Following FL-bench: clients return model_params_diff (negative of update)
        total_samples = sum(result['samples'] for result in client_results)
        weights = torch.tensor([result['samples'] / total_samples for result in client_results], dtype=torch.float32)
        
        # Extract model differences from client results
        params_diff_list = []
        for result in client_results:
            if 'model_params_diff' in result:
                # Client returned differences directly (FL-bench style: diff = global - local)
                # We need to negate to get the update direction
                diff_dict = result['model_params_diff']
                diff_list = [-diff_dict[key] for key in model_keys]  # Negate to get update direction
                params_diff_list.append(diff_list)
            else:
                # Fallback: compute delta from model_state
                global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
                diff_list = []
                for key in model_keys:
                    local_tensor = result['model_state'][key].float()
                    delta = local_tensor - global_snapshot[key]
                    diff_list.append(delta)
                params_diff_list.append(diff_list)
        
        # Compute weighted average of differences
        # Following FL-bench: params_diff = sum(weights * diffs)
        params_diff = []
        # Transpose params_diff_list: from [client][param] to [param][client]
        diffs_by_param = list(zip(*params_diff_list))
        
        for param_idx in range(len(model_keys)):
            # Get diffs for this parameter from all clients
            diffs_for_param = list(diffs_by_param[param_idx])
            
            # Stack along last dimension: shape becomes [..., num_clients]
            # Following FL-bench: torch.stack(diffs, dim=-1) * weights
            diffs_stacked = torch.stack(diffs_for_param, dim=-1)
            
            # Weighted sum: sum(weights * diffs) along last dimension
            # weights has shape [num_clients], broadcasts correctly
            weighted_diff = torch.sum(diffs_stacked * weights, dim=-1)
            params_diff.append(weighted_diff)
        
        # Update momentums: m = beta1 * m + (1 - beta1) * diff
        with torch.no_grad():
            for m, diff in zip(self.momentums, params_diff):
                m.data.mul_(self.beta1).add_((1 - self.beta1) * diff)
            
            # Update velocities according to optimizer type
            for v, diff in zip(self.velocities, params_diff):
                self._update_velocity(v, diff)
            
            # Update model parameters: param += lr * (m / (sqrt(v) + tau))
            for param, m, v in zip(global_state.values(), self.momentums, self.velocities):
                # Compute sqrt(v) + tau
                v_sqrt = v.sqrt().add_(self.tau)
                
                # Compute update: lr * (m / v_sqrt)
                update = (self.server_lr * (m / v_sqrt)).to(param.dtype)
                param.data.add_(update)
        
        global_model.load_state_dict(global_state)
        return global_model

class NormClippingAggregation(BaseAggregation):
    """Norm clipping aggregation with optional weak-DP noise.
    [Can You Really Backdoor Federated Learning](https://arxiv.org/abs/1911.07963) - NeurIPS '20
    It clips the norm of each client gradient updates by a threshold

    Clips the L2 norm of each client's update to a threshold before averaging.
    Config keys:
      - norm_threshold: float (default 3.0)
      - weakDP: bool (default False)
      - noise_mean: float (default 0.0)
      - noise_std: float (default 0.002)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
         # fixed threshold is not good setting, use median of the norms of the client updates is better
        self.norm_threshold = float(self.params.get('norm_threshold', 3.0))
        self.weak_dp = bool(self.params.get('weakDP', False))
        self.noise_mean = float(self.params.get('noise_mean', 0.0))
        self.noise_std = float(self.params.get('noise_std', 0.002))

    def _clip_update_in_place(self, update: Dict[str, torch.Tensor], threshold: float) -> None:
        # Compute L2 norm over the concatenated update; scale if above threshold
        # Work in float32 to avoid overflows/underflows across dtypes
        flat_list = []
        for tensor in update.values():
            flat_list.append(tensor.detach().float().view(-1))
        if not flat_list:
            return
        flat = torch.cat(flat_list)
        norm = torch.norm(flat, p=2).clamp(min=1e-12)
        # print(f"🔍 NormClippingAggregation: norm: {norm.item()}")
        # this norm might have a little bit different with the norm of the global model and client update compute_l2_distance
        if norm.item() > 0 and norm.item() > threshold:
            scale = threshold / norm
            for key in update:
                update[key] = update[key].detach().float() * scale
            # flat_update = torch.cat([update[key].view(-1) for key in update])
            # norm_update = torch.norm(flat_update, p=2)
            # print(f"🔍 NormClippingAggregation: norm_update: {norm_update.item()}")

    def _maybe_add_noise_in_place(self, update: Dict[str, torch.Tensor]) -> None:
        if not self.weak_dp or self.noise_std <= 0:
            return
        for key, tensor in update.items():
            # Skip batch normalization tracking parameters
            if any(s in key for s in ['running_mean', 'running_var', 'num_batches_tracked']):
                continue
            noise = torch.randn_like(tensor.detach().float()) * self.noise_std + self.noise_mean
            update[key] = tensor.detach().float() + noise

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int, verbose: bool = True) -> nn.Module:
        if not client_results:
            return global_model

        total_samples = sum(result['samples'] for result in client_results)
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()

        if verbose:
            # print L2 distance between each client and global model
            self.print_l2_distance(global_model, client_results)

        # Snapshot of global for delta computation in float32
        global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Prepare accumulator
        aggregated_update = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_snapshot.items()}

        # change the threshold to median of the norms of the client updates
        # have to change to median because the fixed norm is too small for some clients
        norms = [compute_l2_distance(result['model_state'], global_model.state_dict()) for result in client_results ]
        self.norm_threshold = torch.median(torch.tensor(norms)).item()
        print(f"🔍 NormClippingAggregation: norm_threshold: {self.norm_threshold}")

        # Build, clip, noise, and weight each client's update
        with torch.no_grad():
            for result in client_results:
                weight = result['samples'] / total_samples
                local_state = result['model_state']

                # Compute client's raw update (delta)
                client_update = {}
                for key in model_keys:
                    client_update[key] = local_state[key].detach().float() - global_snapshot[key]

                # Clip by L2 norm
                self._clip_update_in_place(client_update, self.norm_threshold)

                # Optional weak-DP noise after clipping
                self._maybe_add_noise_in_place(client_update)

                # Accumulate weighted update
                for key in model_keys:
                    aggregated_update[key] += float(weight) * client_update[key]

            # Apply aggregated update to global model (cast back to original dtypes)
            for key in model_keys:
                if global_state[key].dtype != aggregated_update[key].dtype:
                    aggregated_update[key] = aggregated_update[key].to(global_state[key].dtype)
                global_state[key].add_(aggregated_update[key])

        global_model.load_state_dict(global_state)
        return global_model


class CRFLAggregation(BaseAggregation):
    """CRFL: clip-and-noise on the mean aggregated update.
    [CRFL: Certifiably Robust Federated Learning against Backdoor Attacks](http://proceedings.mlr.press/v139/xie21a/xie21a.pdf)

    CRFL apply parameters clipping and perturbing to mean aggregated update
    Steps per round:
      1) Compute each client's delta (local - global_snapshot).
      2) Coordinate-wise mean of deltas to get one aggregated delta.
      3) Clip the aggregated delta's L2 norm to `norm_threshold`.
      4) Optionally add Gaussian noise (mean `noise_mean`, std `noise_std`).
      5) Apply aggregated delta to the global model.

    Config keys (under params):
      - norm_threshold: float (default 3.0)
      - noise_mean: float (default 0.0)
      - noise_std: float (default 0.001)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.name = config.get('name', self.__class__.__name__) # CRFL
        print(f"🔍 CRFLAggregation: params: {self.params}")
        self.norm_threshold = float(self.params.get('norm_threshold', 3.0))
        self.noise_mean = float(self.params.get('noise_mean', 0.0))
        self.noise_std = float(self.params.get('noise_std', 0.001))

    def _flat_concat(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        flat = [t.detach().float().view(-1) for t in tensors]
        return torch.cat(flat) if len(flat) else torch.tensor(0.0)

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        # Float snapshot for delta computation
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # change the threshold to median of the norms of the client updates
        norms = [compute_l2_distance(result['model_state'], global_model.state_dict()) for result in client_results ]
        self.norm_threshold = torch.median(torch.tensor(norms)).item()
        print(f"🔍 CRFLAggregation: norm_threshold: {self.norm_threshold}")

        # 1) coordinate-wise mean delta across clients
        with torch.no_grad():
            agg_delta: Dict[str, torch.Tensor] = {}
            for key in model_keys:
                stacked = torch.stack([cr['model_state'][key].detach().float() - snapshot[key]
                                       for cr in client_results], dim=0)
                agg_delta[key] = stacked.mean(dim=0)

            # 2) clip aggregated delta by global L2 norm threshold
            flat = self._flat_concat([agg_delta[k] for k in model_keys])
            # Guard against empty tensors
            if flat.numel() > 0:
                norm = torch.norm(flat, p=2).clamp(min=1e-12)
                if norm.item() > 0 and norm.item() > self.norm_threshold:
                    scale = self.norm_threshold / norm
                    for k in model_keys:
                        agg_delta[k] = agg_delta[k] * scale

            # 3) add Gaussian noise (optional)
            if self.noise_std > 0:
                for k in model_keys:
                    # Skip batch normalization tracking parameters
                    if any(s in k for s in ['running_mean', 'running_var', 'num_batches_tracked']):
                        continue
                    noise = torch.randn_like(agg_delta[k]) * self.noise_std + self.noise_mean
                    agg_delta[k] = agg_delta[k] + noise

            # 4) apply to global model (cast back to original dtype)
            for key in model_keys:
                global_state[key].add_(agg_delta[key].to(global_state[key].dtype))

        global_model.load_state_dict(global_state)
        return global_model

# class KrumLikeAggregation(BaseAggregation):
#     """
#     [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17
#     Multi-Krum is a variant of Krum that selects the m updates with the smallest scores, rather than just the single update chosen by Krum, where the score is the sum of the n-f-1 smallest Euclidean distances to the other updates. Then it verages these selected updates to produce the final aggregated update.
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.name = config.get('name', self.__class__.__name__) # Krum or MultiKrum
#         # if name is MultiKrum, then avg_percentage is the percentage of clients to be selected for averaging (set = 0.2)
#         # else then select avg_percentage as the number of clients to be selected for averaging (set = 1)
#         self.params = config.get('params', {})
#         self.f = int(self.params.get('f', 0)) # number of Byzantine clients
#         # avg_percentage = 1 => Krum, avg_percentage = 0.2 (< 1) => MultiKrum
#         self.avg_percentage = float(self.params.get('avg_percentage', 0.2))
#
#     def _pairwise_sq_dists(self, vectors):
#         X = torch.stack(vectors).float()
#         xx = (X * X).sum(dim=1, keepdim=True)
#         return (xx + xx.t() - 2 * (X @ X.t())).clamp_min(0)
#
#     def aggregate(self, global_model, client_results, round_num):
#         if not client_results:
#             return global_model
#         n = len(client_results)
#         f = min(self.f, max(0, n - 2))
#         # m: absolute or percentage
#         m = max(1, int(self.avg_percentage * n)) if self.avg_percentage < 1 else int(self.avg_percentage)
#         # m = 1 => Krum, m < 1 => MultiKrum (get partial of clients)
#         # print(f"🔍 KrumLikeAggregation: m: {m} avg_percentage: {self.avg_percentage}")
#         model_keys = list(client_results[0]['model_state'].keys())
#         global_state = global_model.state_dict()
#         snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
#
#         updates = []
#         for cr in client_results:
#             flat = []
#             for k in model_keys:
#                 flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
#             updates.append(torch.cat(flat))
#
#         dists = self._pairwise_sq_dists(updates)
#         scores = []
#         for i in range(n):
#             # Exclude self-distance (zero) and take the (n-f-2) closest others
#             sorted_row = torch.sort(dists[i])[0]
#             num_closest = max(1, n - f - 2)
#             vals = sorted_row[1:num_closest + 1]
#             scores.append(vals.sum())
#         order = torch.argsort(torch.stack(scores))[:m]
#         print(f"🔍 KrumLikeAggregation: order: {order}")
#
#         if m == 1:
#             chosen = updates[int(order[0])]
#         else:
#             # Weighted average by num_examples of selected clients
#             selected_indices = [int(i) for i in order]
#             selected_updates = [updates[idx] for idx in selected_indices]
#             selected_weights = [float(client_results[idx]['samples']) for idx in selected_indices]
#             weight_sum = sum(selected_weights) if sum(selected_weights) > 0 else 1.0
#             normalized = [w / weight_sum for w in selected_weights]
#             stacked_sel = torch.stack(selected_updates)
#             weights_tensor = torch.tensor(normalized, dtype=stacked_sel.dtype, device=stacked_sel.device).unsqueeze(1)
#             chosen = (weights_tensor * stacked_sel).sum(dim=0)
#         # if using mean aggregation, then use the following code
#         # chosen = updates[int(order[0])] if m == 1 else torch.stack([updates[int(i)] for i in order]).mean(dim=0)
#         with torch.no_grad():
#             offset = 0
#             for key in model_keys:
#                 numel = snapshot[key].numel()
#                 delta = chosen[offset:offset + numel].view_as(snapshot[key])
#                 offset += numel
#                 global_state[key].add_(delta.to(global_state[key].dtype))
#         global_model.load_state_dict(global_state)
#         return global_model


class KrumLikeAggregation(BaseAggregation):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.get('name', self.__class__.__name__)
        self.params = config.get('params', {})
        self.f = int(self.params.get('f', 0))
        self.avg_percentage = float(self.params.get('avg_percentage', 0.2))

    def _pairwise_sq_dists(self, vectors):
        X = torch.stack(vectors).float()
        xx = (X * X).sum(dim=1, keepdim=True)
        return (xx + xx.t() - 2 * (X @ X.t())).clamp_min(0)

    def aggregate(self, global_model, client_results, round_num):
        if not client_results:
            return global_model
        n = len(client_results)
        f = min(self.f, max(0, n - 2))
        m = max(1, int(self.avg_percentage * n)) if self.avg_percentage < 1 else int(self.avg_percentage)

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        updates = []
        for cr in client_results:
            flat = []
            for k in model_keys:
                flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
            updates.append(torch.cat(flat))

        dists = self._pairwise_sq_dists(updates)
        scores = []
        for i in range(n):
            sorted_row = torch.sort(dists[i])[0]
            num_closest = max(1, n - f - 2)
            vals = sorted_row[1:num_closest + 1]
            scores.append(vals.sum())
        order = torch.argsort(torch.stack(scores))[:m]
        print(f"🔍 KrumLikeAggregation: order: {order}")

        # ======= 【新增代码】记录本次被选中的客户端 ID =======
        self.last_accepted_clients = [client_results[int(i)].get('client_id', -1) for i in order]
        # ===================================================

        if m == 1:
            chosen = updates[int(order[0])]
        else:
            selected_indices = [int(i) for i in order]
            selected_updates = [updates[idx] for idx in selected_indices]
            selected_weights = [float(client_results[idx]['samples']) for idx in selected_indices]
            weight_sum = sum(selected_weights) if sum(selected_weights) > 0 else 1.0
            normalized = [w / weight_sum for w in selected_weights]
            stacked_sel = torch.stack(selected_updates)
            weights_tensor = torch.tensor(normalized, dtype=stacked_sel.dtype, device=stacked_sel.device).unsqueeze(1)
            chosen = (weights_tensor * stacked_sel).sum(dim=0)

        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = chosen[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)
        return global_model

class MedianAggregation(BaseAggregation):
    """Median aggregation over raw client updates (no delta).
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Coordinated Median computes the median of the updates coordinate-wisely.
    """

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        if not client_results:
            return global_model
        
        client_states = [result['model_state'] for result in client_results]
        median_state = {}
        
        with torch.no_grad():
            for name in global_model.state_dict():
                stacked = torch.stack([client_state[name].float() for client_state in client_states], dim=0)
                median = torch.median(stacked, dim=0)[0]
                median_state[name] = median.to(global_model.state_dict()[name].dtype)
        
        global_model.load_state_dict(median_state)
        return global_model


class CoordinateWiseMedianAggregation(BaseAggregation):
    """Coordinate-wise Median over client deltas (local - global).
    [Byzantine-robust distributed learning: Towards optimal statistical rates] - ICML'18

    This matches FLPoison's Median: compute per-coordinate median of client updates
    (deltas), then apply the aggregated delta to the global model.
    """

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        median_delta: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key in model_keys:
                stacked = torch.stack([
                    client['model_state'][key].detach().float() - snapshot[key]
                    for client in client_results
                ], dim=0)
                median = torch.median(stacked, dim=0)[0]
                median_delta[key] = median

            # Apply delta to global model (cast to original dtype)
            for key in model_keys:
                global_state[key].add_(median_delta[key].to(global_state[key].dtype))

        global_model.load_state_dict(global_state)
        return global_model



class RFAAggregation(BaseAggregation):
    """RFA (Geometric Median) aggregation via smoothed Weiszfeld algorithm.

    Config keys:
      - num_iters: int (default 3)
      - epsilon: float (default 1e-6)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get('name', self.__class__.__name__) # RFA
        self.params = config.get('params', {})  
        self.num_iters = int(self.params.get('num_iters', 3))
        self.epsilon = float(self.params.get('epsilon', 1.0e-6))
        print(f"🔍 RFAAggregation: num_iters: {self.num_iters} epsilon: {self.epsilon}")

    def _stack_client_updates(self, model_keys: List[str], global_snapshot: Dict[str, torch.Tensor], client_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Build a flat vector delta for a single client in float32
        flat_list = []
        for key in model_keys:
            delta = client_state[key].detach().float() - global_snapshot[key]
            flat_list.append(delta.view(-1))
        return torch.cat(flat_list)

    def _smoothed_weiszfeld(self, updates_2d: torch.Tensor) -> torch.Tensor:
        # updates_2d: shape (num_clients, num_params)
        device = updates_2d.device
        v = torch.zeros(updates_2d.size(1), dtype=torch.float32, device=device)
        alphas = torch.full((updates_2d.size(0),), 1.0 / updates_2d.size(0), dtype=torch.float32, device=device)
        for _ in range(self.num_iters):
            denom = torch.norm(updates_2d - v.unsqueeze(0), p=2, dim=1)
            betas = alphas / torch.clamp(denom, min=self.epsilon)
            v = (betas.unsqueeze(1) * updates_2d).sum(dim=0) / betas.sum()
        return v

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Stack client deltas into a 2D tensor [num_clients, num_params]
        flat_updates = []
        for result in client_results:
            flat_updates.append(self._stack_client_updates(model_keys, global_snapshot, result['model_state']))
        updates_2d = torch.stack(flat_updates, dim=0)

        # Geometric median of client updates
        gm = self._smoothed_weiszfeld(updates_2d)

        # Unflatten and apply to global
        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = global_snapshot[key].numel()
                delta = gm[offset:offset + numel].view_as(global_snapshot[key])
                offset += numel
                delta = delta.to(global_state[key].dtype)
                global_state[key].add_(delta)

        global_model.load_state_dict(global_state)
        return global_model


class CenteredClippingAggregation(BaseAggregation):
    """Centered Clipping: clip updates relative to momentum history.
    [Learning from History for Byzantine Robust Optimization](https://arxiv.org/abs/2012.10333) - ICML '21
    
    Clip client updates relative to momentum history instead of global model.
    Momentum accumulates across rounds to provide adaptive filtering.
    
    Steps per round:
      1) Compute difference between each client update and current momentum.
      2) Clip these differences by L2 norm threshold.
      3) Average clipped differences and add to momentum (iteratively refine).
      4) Apply updated momentum to global model.
    
    Config keys (under params):
      - norm_threshold: float (default -1.0, -1 = adaptive, >0 = fixed threshold)
      - num_iters: int (default 1)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.norm_threshold_param = float(self.params.get('norm_threshold', -1.0))  # -1 means adaptive
        self.num_iters = int(self.params.get('num_iters', 1))
        self.momentum: torch.Tensor = None  # Persistent momentum state
        print(f"🔍 CenteredClippingAggregation: norm_threshold_param={self.norm_threshold_param}, num_iters={self.num_iters}")
    
    def _clip_update(self, diff: torch.Tensor) -> torch.Tensor:
        """Clip update by L2 norm threshold."""
        norm = torch.norm(diff, p=2).clamp(min=1e-12)
        scale = min(1.0, (self.norm_threshold / norm).item())
        return diff * scale
    
    def _stack_update(self, model_keys: List[str], client_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack client updates into flat tensor."""
        flat_list = []
        for key in model_keys:
            flat_list.append(client_state[key].detach().float().view(-1))
        return torch.cat(flat_list)
    
    def _unstack_update(self, flat_update: torch.Tensor, model_keys: List[str], global_snapshot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unstack flat tensor back into model update dictionary."""
        update_dict = {}
        offset = 0
        for key in model_keys:
            numel = global_snapshot[key].numel()
            update_dict[key] = flat_update[offset:offset + numel].view_as(global_snapshot[key])
            offset += numel
        return update_dict
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        global_snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        
        # Stack client updates into flat tensors
        client_updates_flat = []
        for cr in client_results:
            client_updates_flat.append(self._stack_update(model_keys, cr['model_state']))
        U = torch.stack(client_updates_flat)  # [n_clients, n_params]
        
        # Initialize momentum if first round
        if self.momentum is None:
            self.momentum = torch.zeros_like(U[0])
        
        # Determine threshold: adaptive if param < 0
        if self.norm_threshold_param < 0:
            # Adaptive threshold: use median norm of differences from momentum
            if self.momentum is not None:
                diffs = U - self.momentum.unsqueeze(0)
                norms = torch.norm(diffs, p=2, dim=1)
                self.norm_threshold = float(torch.median(norms).item())
            else:
                # First round: use median norm of client updates
                norms = torch.norm(U, p=2, dim=1)
                self.norm_threshold = float(torch.median(norms).item())
            print(f"🔍 CenteredClippingAggregation: using adaptive threshold={self.norm_threshold:.4f}")
        else:
            # Use fixed threshold from config
            self.norm_threshold = self.norm_threshold_param
        
        # Iterative refinement: clip and update momentum
        for _ in range(self.num_iters):
            # Compute differences from momentum
            diffs = U - self.momentum.unsqueeze(0)  # [n_clients, n_params]
            
            # Clip differences
            clipped_diffs = torch.stack([self._clip_update(diff) for diff in diffs])
            
            # Average clipped differences
            avg_clipped_diff = clipped_diffs.mean(dim=0)
            
            # Update momentum
            self.momentum = self.momentum + avg_clipped_diff
        
        # Apply momentum update to global model
        with torch.no_grad():
            momentum_dict = self._unstack_update(self.momentum, model_keys, global_snapshot)
            for key in model_keys:
                global_state[key].add_(momentum_dict[key].to(global_state[key].dtype))
        
        global_model.load_state_dict(global_state)
        return global_model


class TrimmedMeanAggregation(BaseAggregation):
    """Coordinate-wise Trimmed Mean over client deltas (local - global).
    [Byzantine-robust distributed learning: Towards optimal statistical rates](https://proceedings.mlr.press/v80/yin18a.html) - ICML'18
    Trimmed Mean exludes the smallest and largest beta fraction coordiantes of the updates and averages the rest coordiantes.
    
    Config keys:
      - proportion: float in [0, 0.5) - fraction of clients to exclude from the trimmed mean
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get('name', self.__class__.__name__) # TrimmedMean
        self.params = config.get('params', {})
        self.proportion = float(self.params.get('proportion', 0.1))
        print(f"🔍 TrimmedMeanAggregation: proportion: {self.proportion}")

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        proportion = max(0.0, min(self.proportion, 0.499999))
        print(f"🔍 TrimmedMeanAggregation: proportion: {proportion}")

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        trimmed_delta: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key in model_keys:
                # Stack client deltas for this parameter: shape [n_clients, ...]
                stacked = torch.stack([
                    client['model_state'][key].detach().float() - snapshot[key]
                    for client in client_results
                ], dim=0)
                num_clients = stacked.size(0)
                k = int(proportion * num_clients)
                k = max(0, min(k, (num_clients - 1) // 2))
                # Sort per coordinate and trim
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[k:num_clients - k].mean(dim=0)
                trimmed_delta[key] = trimmed

            # Apply delta to global model (cast to original dtype)
            for key in model_keys:
                global_state[key].add_((trimmed_delta[key]).to(global_state[key].dtype))

        global_model.load_state_dict(global_state)
        return global_model





class SimpleClusteringAggregation(BaseAggregation):
    """Two-cluster k-means; average from majority cluster."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_iters = int(self.config.get('max_iters', 5))

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        updates = []
        for cr in client_results:
            flat = []
            for k in model_keys:
                flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
            updates.append(torch.cat(flat))
        X = torch.stack(updates)

        n = X.size(0)
        idx = torch.randperm(n)[:2]
        centroids = X[idx].clone()
        for _ in range(self.max_iters):
            dists = torch.cdist(X, centroids, p=2)
            labels = torch.argmin(dists, dim=1)
            new_centroids = []
            for c in range(2):
                mask = labels == c
                if mask.any():
                    new_centroids.append(X[mask].mean(dim=0))
                else:
                    new_centroids.append(centroids[c])
            new_centroids = torch.stack(new_centroids)
            if torch.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        majority = 0 if (labels == 0).sum() >= (labels == 1).sum() else 1
        gm = X[labels == majority].mean(dim=0)

        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = gm[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)
        return global_model


class FlameAggregation(BaseAggregation):
    """
    FLAME: Taming Backdoors in Federated Learning (USENIX Security '22)
    FLAME: cluster by cosine (HDBSCAN), clip by median norm, add DP noise.
    Mirrors FLPoison FLAME behavior using client deltas as updates.
    
    Config keys (under params):
      - gamma: float (default 1.2e-5) noise scaling factor
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.gamma = float(self.params.get('gamma', 1.2e-5))

    def _stack_updates(self, model_keys: List[str], snapshot: Dict[str, torch.Tensor], client_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat_list = []
        for key in model_keys:
            delta = client_state[key].detach().float() - snapshot[key]
            flat_list.append(delta.view(-1))
        return torch.cat(flat_list)

    def _normclip(self, U: torch.Tensor, threshold: float) -> torch.Tensor:
        norms = torch.norm(U, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = (threshold / norms).clamp(max=1.0)
        return U * scale

    def _add_noise_to_model(self, model: nn.Module, noise_scale: float) -> None:
        from copy import deepcopy
        with torch.no_grad():
            state = deepcopy(model.state_dict())
            for key, tensor in state.items():
                if any(s in key for s in ['running_mean', 'running_var', 'num_batches_tracked']):
                    continue
                std = tensor.detach().float().std()
                if not torch.isfinite(std):
                    std = torch.tensor(0.0, dtype=torch.float32, device=tensor.device)
                noise = torch.normal(mean=0.0, std=(noise_scale * std).item() if std.numel() == 1 else noise_scale * std, size=tensor.size(), device=tensor.device)
                tensor.add_(noise.to(tensor.dtype))
            model.load_state_dict(state)

    # def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
    #     if not client_results:
    #         return global_model
    #
    #     # print l2 norm of the global model
    #     global_model_params = list(global_model.parameters())
    #     global_model_norm = torch.norm(torch.cat([p.view(-1) for p in global_model_params]), p=2)
    #     print(f"🔍 FlameAggregation: before clustering, global_model_norm: {global_model_norm}")
    #
    #     model_keys = list(client_results[0]['model_state'].keys())
    #     global_state = global_model.state_dict()
    #     snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
    #
    #     # print id of all clients
    #     print(f"🔍 FlameAggregation: client_id: {[cr['client_id'] for cr in client_results]}")
    #     # Build per-client flat model/gradient updates (same here as deltas)
    #     updates = []
    #     for cr in client_results:
    #         updates.append(self._stack_updates(model_keys, snapshot, cr['model_state']))
    #     U = torch.stack(updates)  # [n, P]
    #
    #     # HDBSCAN clustering on cosine distances over model updates
    #     try:
    #         import hdbscan  # type: ignore
    #     except Exception as e:
    #         raise ImportError("hdbscan is required for FlameAggregation") from e
    #     X = U.detach().cpu().numpy().astype(np.float64)
    #     cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
    #                               min_cluster_size=max(2, U.size(0)//2+1), min_samples=1, allow_single_cluster=True)
    #     cluster.fit(X)
    #     benign_idx = [idx for idx, label in enumerate(cluster.labels_) if label == 0]
    #     if len(benign_idx) == 0:
    #         benign_idx = list[int](range(U.size(0)))
    #
    #     print(f"🔍 FlameAggregation: benign_idx: {benign_idx} and cluster.labels_: {cluster.labels_}")
    #     # import IPython; IPython.embed();
    #     # Median L2 norm across all clients
    #     norms = torch.norm(U, p=2, dim=1)
    #     median_norm = float(torch.median(norms).item())
    #     print(f"🔍 FlameAggregation: median_norm: {median_norm} and norms: {norms}")
    #
    #     # Clip benign gradients by median norm, then average
    #     clipped = self._normclip(U[benign_idx], median_norm)
    #     agg_grad = clipped.mean(dim=0)
    #     # print(f"🔍 FlameAggregation: U[benign_idx].shape: {U[benign_idx].shape}")
    #     # Big question: why filter out all attacks client but the backdoor accuracy is still high?
    #     # 10 clients:  [3, 14, 81, 94, 29, 74, 95, 42, 86, 65]
    #     # first four are adversarial clients: [3, 14, 81, 94]
    #     # last six are benign clients: [29, 74, 95, 42, 86, 65]
    #     # cluster.labels_: array([-1, -1, -1, -1,  0,  0,  0,  0,  0,  0])
    #     # so why the backdoor accuracy is still high?
    #     # because the benign clients are still able to learn the backdoor?
    #     # or because the adversarial clients are not able to learn the backdoor?
    #     # or because the adversarial clients are not able to learn the backdoor?
    #     # Apply aggregated update
    #     with torch.no_grad():
    #         offset = 0
    #         for key in model_keys:
    #             numel = snapshot[key].numel()
    #             delta = agg_grad[offset:offset + numel].view_as(snapshot[key])
    #             offset += numel
    #             global_state[key].add_(delta.to(global_state[key].dtype))
    #     global_model.load_state_dict(global_state)
    #
    #     # Add DP-like Gaussian noise scaled by gamma * median_norm
    #     self._add_noise_to_model(global_model, self.gamma * median_norm)
    #     # print l2 norm of the global model
    #     global_model_params = list(global_model.parameters())
    #     global_model_norm = torch.norm(torch.cat([p.view(-1) for p in global_model_params]), p=2)
    #     print(f"🔍 FlameAggregation: after adding noise, global_model_norm: {global_model_norm}")
    #     return global_model
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        updates = []
        for cr in client_results:
            updates.append(self._stack_updates(model_keys, snapshot, cr['model_state']))
        U = torch.stack(updates)

        import hdbscan
        import numpy as np
        X = U.detach().cpu().numpy().astype(np.float64)
        cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                                  min_cluster_size=max(2, U.size(0) // 2 + 1), min_samples=1, allow_single_cluster=True)
        cluster.fit(X)
        benign_idx = [idx for idx, label in enumerate(cluster.labels_) if label == 0]
        if len(benign_idx) == 0:
            benign_idx = list[int](range(U.size(0)))

        # ======= 【新增代码】记录本次被选中的客户端 ID =======
        self.last_accepted_clients = [client_results[i].get('client_id', -1) for i in benign_idx]
        # ===================================================

        norms = torch.norm(U, p=2, dim=1)
        median_norm = float(torch.median(norms).item())

        clipped = self._normclip(U[benign_idx], median_norm)
        agg_grad = clipped.mean(dim=0)

        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = agg_grad[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)

        self._add_noise_to_model(global_model, self.gamma * median_norm)
        return global_model


class DeepSightAggregation(BaseAggregation):
    """DeepSight: filter clients via NEUP/TEs, DDifs, cosine on biases; clip and aggregate.
    Mirrors FLPoison DeepSight with configurable params.
    
    Config keys (under params):
      - num_seeds: int (default 3)
      - threshold_factor: float (default 0.01)
      - num_samples: int (default 20000)
      - tau: float (default 0.33)
      - epsilon: float (default 1e-6)
      - num_channels: int, num_dims: int, batch_size: int, num_workers: int, device: str
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.num_seeds = int(self.params.get('num_seeds', 3))
        self.threshold_factor = float(self.params.get('threshold_factor', 0.01))
        self.num_samples = int(self.params.get('num_samples', 20000))
        self.tau = float(self.params.get('tau', 0.33))
        self.epsilon = float(self.params.get('epsilon', 1.0e-6))
        self.num_channels = int(self.params.get('num_channels', 3))
        self.num_dims = int(self.params.get('num_dims', 32))
        self.batch_size = int(self.params.get('batch_size', 128))
        self.num_workers = int(self.params.get('num_workers', 0))
        self.device = self.params.get('device', 'gpu')

    def _stack_updates(self, model_keys: List[str], snapshot: Dict[str, torch.Tensor], client_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat_list = []
        for key in model_keys:
            delta = client_state[key].detach().float() - snapshot[key]
            flat_list.append(delta.view(-1))
        return torch.cat(flat_list)

    def _get_last_linear_updates(self, global_snapshot: Dict[str, torch.Tensor], client_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        last_w_name, last_b_name = None, None
        for name, tensor in global_snapshot.items():
            if tensor.dim() == 2:
                last_w_name = name
        if last_w_name is not None:
            base = last_w_name.rsplit('.', 1)[0]
            bname = base + '.bias'
            if bname in global_snapshot:
                last_b_name = bname
        if last_w_name is None or last_b_name is None:
            # Fallback: no identifiable linear layer; return zeros
            return {'weight': torch.zeros(1, 1), 'bias': torch.zeros(1)}
        w_delta = client_state[last_w_name].detach().float() - global_snapshot[last_w_name]
        b_delta = client_state[last_b_name].detach().float() - global_snapshot[last_b_name]
        return {'weight': w_delta, 'bias': b_delta}

    def _compute_NEUPs_TEs(self, ol_updates_list: List[Dict[str, torch.Tensor]]) -> (torch.Tensor, torch.Tensor):
        num_clients = len(ol_updates_list)
        bias0 = ol_updates_list[0]['bias']
        num_classes = bias0.numel()
        NEUPs = torch.empty((num_clients, num_classes), dtype=torch.float32)
        TEs = torch.empty((num_clients,), dtype=torch.float32)
        threshold_factor = max(self.threshold_factor, 1.0 / float(num_classes))
        for cid in range(num_clients):
            upd = ol_updates_list[cid]
            # update_energy per class: |bias| + sum(|weight_row|)
            energy = upd['bias'].abs() + upd['weight'].abs().sum(dim=1)
            energy_sq = energy.pow(2)
            NEUP = energy_sq / (energy_sq.sum() + self.epsilon)
            threshold = threshold_factor * NEUP.max()
            TEs[cid] = (NEUP > threshold).sum()
            NEUPs[cid] = NEUP
        return NEUPs, TEs

    def _cosine_dist_bias(self, ol_updates_list: List[Dict[str, torch.Tensor]]):
        from sklearn.metrics.pairwise import cosine_distances
        bias_mat = torch.stack([u['bias'].view(-1) for u in ol_updates_list], dim=0).detach().cpu().numpy()
        return cosine_distances(bias_mat.reshape(bias_mat.shape[0], -1)).astype(np.float64)

    def _gen_randdata(self):
        # Pre-generate random dataset tensors for time saving
        class RandDataset(torch.utils.data.Dataset):
            def __init__(self, size, num_samples, seed):
                self.num_samples = num_samples
                torch.manual_seed(seed)
                self.dataset = torch.rand(num_samples, *size)
            def __len__(self):
                return self.num_samples
            def __getitem__(self, idx):
                return self.dataset[idx]
        size = [self.num_channels, self.num_dims, self.num_dims]
        return [RandDataset(size, self.num_samples, seed) for seed in range(self.num_seeds)]

    def _cluster_labels(self, cosine_dists, NEUPs_np, DDifs_np):
        import hdbscan
        def cluster_dists(statistic, precomputed=False):
            func = hdbscan.HDBSCAN(min_samples=3, metric='precomputed') if precomputed else hdbscan.HDBSCAN(min_samples=3)
            labels = func.fit_predict(statistic)
            # distance matrix from cluster labels: 0 if same cluster else 1
            same = (labels[:, None] == labels)
            return np.where(same, 0, 1)
        cosine_cluster_dists = cluster_dists(cosine_dists, precomputed=True)
        neup_cluster_dists = cluster_dists(NEUPs_np)
        ddif_cluster_dists = np.array([cluster_dists(DDifs_np[i]) for i in range(self.num_seeds)])
        merged_ddif = ddif_cluster_dists.mean(axis=0)
        merged = np.mean([merged_ddif, neup_cluster_dists, cosine_cluster_dists], axis=0)
        labels = hdbscan.HDBSCAN(metric='precomputed', allow_single_cluster=True, min_samples=3).fit_predict(merged)
        return labels

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Build client flat updates and output-layer updates
        updates = []
        ol_updates_list: List[Dict[str, torch.Tensor]] = []
        for cr in client_results:
            updates.append(self._stack_updates(model_keys, snapshot, cr['model_state']))
            ol_updates_list.append(self._get_last_linear_updates(snapshot, cr['model_state']))
        U = torch.stack(updates)

        # 2) Filtering layer inputs
        NEUPs, TEs = self._compute_NEUPs_TEs(ol_updates_list)
        cosine_dists = self._cosine_dist_bias(ol_updates_list)

        # 3) Ensemble clustering
        # Build client-updated models by applying deltas to global model
        rand_datasets = self._gen_randdata()
        client_models: List[nn.Module] = []
        import copy as _copy
        for cr in client_results:
            m = _copy.deepcopy(global_model).to(self.device)
            # apply client delta to model
            with torch.no_grad():
                for k in model_keys:
                    m.state_dict()[k].add_((cr['model_state'][k].detach().float().to(self.device) - snapshot[k].to(self.device)).to(m.state_dict()[k].dtype))
            client_models.append(m)

        # Compute DDifs per seed per client
        DDifs = []
        # Move a copy of global model to device for inference
        global_model_copy = _copy.deepcopy(global_model).to(self.device)
        for seed in range(self.num_seeds):
            seed_ddifs = []
            loader = torch.utils.data.DataLoader(rand_datasets[seed], self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers, pin_memory=False)
            for cid, m in enumerate(client_models):
                m.eval(); global_model_copy.eval()
                DDif = torch.zeros(NEUPs.size(1))
                seen = 0
                for images in loader:
                    images = images.to(self.device)
                    with torch.no_grad():
                        out_c = m(images)
                        out_g = global_model_copy(images)
                    temp = out_c.detach().cpu() / (out_g.detach().cpu() + self.epsilon)
                    DDif.add_(temp.sum(dim=0))
                    seen += images.size(0)
                seed_ddifs.append((DDif / max(1, seen)).numpy())
            DDifs.append(seed_ddifs)
        DDifs_np = np.array(DDifs)

        # 4) Poisoned cluster identification
        labels = self._cluster_labels(cosine_dists, NEUPs.numpy(), DDifs_np)
        suspicious_flags = TEs <= (NEUPs.new_tensor(np.median(TEs.numpy())) / 2)
        accepted_indices = []
        unique = [l for l in set(labels) if l != -1]
        for lab in unique:
            idxs = np.where(labels == lab)[0]
            amount_susp = int(suspicious_flags[idxs].sum().item())
            if amount_susp < self.tau * len(idxs):
                accepted_indices.extend(list(idxs))
        accepted_indices = np.array(accepted_indices, dtype=np.int64)
        if accepted_indices.size == 0:
            accepted_indices = np.arange(U.size(0), dtype=np.int64)

        # 5) Clip accepted gradient updates w.r.t median L2 norm of all updates
        norms = torch.norm(U, p=2, dim=1)
        median_norm = float(torch.median(norms).item())
        def _normclip(X, thr):
            n = torch.norm(X, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            scl = (thr / n).clamp(max=1.0)
            return X * scl
        clipped = _normclip(U[accepted_indices], median_norm)

        # 6) Aggregate clipped accepted updates and apply
        agg = clipped.mean(dim=0)
        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = agg[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)
        return global_model


class FLTrustAggregation(BaseAggregation):
    """FLTrust: Trust bootstrapping with a server anchor update.
    Approximates FLPoison FLTrust. Requires a root/anchor update or computes a fallback.
    [FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](https://arxiv.org/abs/2012.13995) - NDSS '21
    FLTrust assumes that the server has a small benign dataset and trains a server benign model as the trust anchor, and computes the trust score as the cosine similarity between the client updates and the server models' update. The client updates are normalized by the server models' update, and then weighted by the trust score to compute the final aggregated update.
    
    Config keys (under params):
      - epsilon: float (default 1e-9)
    aggregate kwargs (preferred):
      - root_grad_update: torch.Tensor flat anchor update (optional)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.epsilon = float(self.params.get('epsilon', 1e-9))

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Stack client deltas
        updates = []
        for cr in client_results:
            flat = []
            for k in model_keys:
                flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
            updates.append(torch.cat(flat))
        U = torch.stack(updates)  # [n, P]

        # Anchor/root update
        root = None
        # Preferred: provided by caller
        if 'root_grad_update' in self.params:
            val = self.params['root_grad_update']
            if isinstance(val, torch.Tensor):
                root = val.detach().float().view(-1).to(U.device)
        # Fallback: use mean update as anchor
        if root is None:
            root = U.mean(dim=0)

        # Trust scores via cosine similarity to root (ReLU, normalize)
        rnorm = torch.norm(root, p=2) + self.epsilon
        Unorms = torch.norm(U, p=2, dim=1) + self.epsilon
        sims = (U @ root) / (Unorms * rnorm)
        sims = torch.clamp(sims, min=0.0)
        sum_sims = sims.sum().item()
        if sum_sims <= 0:
            weights = torch.full_like(sims, 1.0 / U.size(0))
        else:
            weights = sims / (sum_sims + self.epsilon)

        # Normalize magnitudes of client updates to ||root||
        target_mag = rnorm.item()
        normed_updates = (U / Unorms.unsqueeze(1)) * target_mag

        # Weighted average
        gm = (weights.unsqueeze(1) * normed_updates).sum(dim=0)

        # Apply to global
        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = gm[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)
        return global_model


class FLDetectorAggregation(BaseAggregation):
    """FLDetector: history-based detection with LBFGS approximation and clustering.
    Mirrors FLPoison FLDetector behavior.

    Config keys (under params):
      - window_size: int (default 10)
      - start_epoch: int (default 50)
    aggregate kwargs:
      - global_epoch: int (required for detection logic)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.window_size = int(self.params.get('window_size', 10))
        self.start_epoch = int(self.params.get('start_epoch', 50))
        # history
        self.global_weight_diffs: List[torch.Tensor] = []
        self.global_grad_diffs: List[torch.Tensor] = []
        self.last_global_grad: torch.Tensor = torch.tensor(0.0)
        self.last_grad_updates: torch.Tensor = torch.tensor(0.0)
        self.malicious_score: List[torch.Tensor] = []

    def _stack_client_updates(self, model_keys: List[str], global_snapshot: Dict[str, torch.Tensor], client_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat_list = []
        for key in model_keys:
            delta = client_state[key].detach().float() - global_snapshot[key]
            flat_list.append(delta.view(-1))
        return torch.cat(flat_list)

    def _lbfgs(self, S_list: List[torch.Tensor], Y_list: List[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        if len(S_list) == 0 or len(Y_list) == 0:
            return torch.zeros_like(v)
        S_k = [s.detach().cpu().numpy().reshape(-1, 1) for s in S_list]
        Y_k = [y.detach().cpu().numpy().reshape(-1, 1) for y in Y_list]
        v_np = v.detach().cpu().numpy().reshape(-1, 1)
        curr_S = np.concatenate(S_k, axis=1)
        curr_Y = np.concatenate(Y_k, axis=1)
        S_t_Y = curr_S.T @ curr_Y
        S_t_S = curr_S.T @ curr_S
        R_k = np.triu(S_t_Y)
        L_k = S_t_Y - R_k
        sigma_k = (Y_k[-1].T @ S_k[-1]) / (S_k[-1].T @ S_k[-1])
        D_diag = np.diag(S_t_Y)
        upper = np.concatenate([sigma_k * S_t_S, L_k], axis=1)
        lower = np.concatenate([L_k.T, -np.diag(D_diag)], axis=1)
        mat = np.concatenate([upper, lower], axis=0)
        mat_inv = np.linalg.inv(mat)
        approx = sigma_k * v_np
        p_mat = np.concatenate([curr_S.T @ (sigma_k * v_np), curr_Y.T @ v_np], axis=0)
        approx -= (np.concatenate([sigma_k * curr_S, curr_Y], axis=1) @ mat_inv) @ p_mat
        return torch.from_numpy(approx.squeeze()).to(v.device, dtype=v.dtype)

    def _gap_statistics(self, data: torch.Tensor, num_sampling: int, K_max: int) -> int:
        from sklearn.cluster import KMeans
        x = data.detach().cpu().numpy()
        x = (x - x.min()) / max(1e-12, (x.max() - x.min()))
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        gaps, s = [], []
        K_max = min(K_max, x.shape[0])
        for k in range(1, K_max + 1):
            kmeans = KMeans(n_clusters=k, n_init=10).fit(x)
            inertia = kmeans.inertia_
            fake_inertia = []
            for _ in range(num_sampling):
                rnd = np.random.rand(x.shape[0], x.shape[1])
                kmeans_fake = KMeans(n_clusters=k, n_init=10).fit(rnd)
                fake_inertia.append(kmeans_fake.inertia_)
            mean_fake = np.mean(fake_inertia)
            gap = np.log(mean_fake) - np.log(inertia)
            gaps.append(gap)
            sd = np.std(np.log(fake_inertia))
            s.append(sd * np.sqrt((1 + num_sampling) / num_sampling))
        num_cluster = 0
        for k in range(1, K_max):
            if gaps[k - 1] - gaps[k] + s[k] >= 0:
                num_cluster = k + 1
                break
        else:
            num_cluster = K_max
        return num_cluster

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int, **kwargs) -> nn.Module:
        if not client_results:
            return global_model

        from sklearn.cluster import KMeans
        current_epoch = int(kwargs.get('global_epoch', 0))

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Build client gradient updates U
        updates = []
        for cr in client_results:
            updates.append(self._stack_client_updates(model_keys, snapshot, cr['model_state']))
        U = torch.stack(updates)  # [n, P]

        benign_idx = np.arange(U.size(0))

        if current_epoch - self.start_epoch > self.window_size and len(self.global_weight_diffs) > 0:
            hvp = self._lbfgs(self.global_weight_diffs, self.global_grad_diffs, self.last_global_grad)
            pred_grad = (self.last_grad_updates if isinstance(self.last_grad_updates, torch.Tensor) else torch.zeros_like(U)) + hvp
            pred = pred_grad.detach().cpu().numpy().reshape(1, -1)
            real = U.detach().cpu().numpy()
            distance = np.linalg.norm(pred - real, axis=1)
            distance = distance / max(1e-12, np.sum(distance))
            self.malicious_score.append(torch.from_numpy(distance))

        if len(self.malicious_score) > self.window_size:
            ms = torch.stack(self.malicious_score[-self.window_size:], dim=0)
            score = ms.mean(dim=0).cpu().numpy()
            if self._gap_statistics(torch.from_numpy(score), num_sampling=20, K_max=10) >= 2:
                estimator = KMeans(n_clusters=2, n_init=10)
                estimator.fit(score.reshape(-1, 1))
                labels = estimator.labels_
                benign_label = 1 if np.mean(score[labels == 0]) > np.mean(score[labels == 1]) else 0
                benign_idx = np.argwhere(labels == benign_label).squeeze()

        agg = U[benign_idx].mean(dim=0)

        # Update histories
        self.global_weight_diffs.append(agg.detach().cpu())
        self.global_grad_diffs.append((agg.detach().cpu() - (self.last_global_grad.detach().cpu() if isinstance(self.last_global_grad, torch.Tensor) else torch.tensor(0.0))))
        if len(self.global_weight_diffs) > self.window_size:
            del self.global_weight_diffs[0]
            del self.global_grad_diffs[0]
        self.last_global_grad = agg.detach()
        self.last_grad_updates = U.detach()

        # Apply
        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = agg[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)
        return global_model

class BulyanAggregation(BaseAggregation):
    """Bulyan: Sequential Krum selection, then β-closest-median coordinate-wise aggregation.
    [The Hidden Vulnerability of Distributed Learning in Byzantium](https://arxiv.org/abs/1802.07927)
    
    Two-stage algorithm:
    1) Sequential Krum selection to build candidate set of size s = n - 2f
    2) β-closest-median coordinate-wise robust aggregation on candidates
    
    Config keys:
      - f: int (default 1) - number of Byzantine clients
      - beta: int (default n-4f) - number of closest-to-median clients for final aggregation
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.f = int(self.config.get('f', 1))
        # β defaults to n-2f, will be set in aggregate()
        self.beta = self.config.get('beta', None)

    def _krum_select_one(self, updates: torch.Tensor, f: int) -> int:
        """Select one client using Krum scoring from remaining updates."""
        n = updates.size(0)
        if n <= 2 * f + 2:
            raise ValueError(f"Krum condition violated: n={n} <= 2f+2={2*f+2}")
        
        dists = self._pairwise_sq_dists([updates[i] for i in range(n)])
        scores = []
        for i in range(n):
            # Exclude self-distance (zero) and take the (n-f-2) closest others
            sorted_row = torch.sort(dists[i])[0]
            num_closest = max(1, n - f - 2)
            vals = sorted_row[1:num_closest + 1]
            scores.append(vals.sum())
        return int(torch.argmin(torch.stack(scores)))

    def _pairwise_sq_dists(self, vectors):
        X = torch.stack(vectors).float()
        xx = (X * X).sum(dim=1, keepdim=True)
        return (xx + xx.t() - 2 * (X @ X.t())).clamp_min(0)

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        n = len(client_results)
        f = min(self.f, max(0, n - 2))
        
        # Set β if not provided: β = (n - 2f) - 2f = n - 4f
        default_beta = max(1, n - 4 * f)
        beta = int(self.beta) if self.beta is not None else default_beta
        beta = max(1, min(beta, n - 2 * f))  # Ensure 1 ≤ β ≤ n - 2f
        
        # Check Bulyan condition: 4f + 3 ≤ n
        if 4 * f + 3 > n:
            raise ValueError(f"Bulyan condition violated: 4f+3={4*f+3} > n={n}")

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        
        # Build client updates
        updates = []
        for cr in client_results:
            flat = []
            for k in model_keys:
                flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
            updates.append(torch.cat(flat))
        updates = torch.stack(updates)

        # Stage 1: Sequential Krum selection to build candidate set
        s = max(1, n - 2 * f)  # candidate set size
        selected_indices = []
        remaining_updates = updates.clone()
        remaining_indices = list(range(n))
        
        while len(selected_indices) < s and len(remaining_indices) > 0:
            try:
                # Select one client using Krum from remaining updates
                selected_idx = self._krum_select_one(remaining_updates, f)
                actual_idx = remaining_indices[selected_idx]
                selected_indices.append(actual_idx)
                
                # Remove selected client from remaining
                remaining_updates = torch.cat([
                    remaining_updates[:selected_idx], 
                    remaining_updates[selected_idx+1:]
                ])
                remaining_indices.pop(selected_idx)
                
            except ValueError:
                # Krum condition violated, stop selection
                break

        if len(selected_indices) == 0:
            raise ValueError("No clients could be selected by Krum")

        # Stage 2: β-closest-median coordinate-wise robust aggregation
        candidates = updates[selected_indices]
        print(f"🔍 BulyanAggregation: beta: {beta} candidates: {len(candidates)} selected_indices: {selected_indices}")
        if beta >= len(candidates):
            # Use all candidates
            aggregated_delta = candidates.mean(dim=0)
        else:
            # Compute coordinate-wise median, then average the β closest-to-median values per coordinate
            median = torch.median(candidates, dim=0)[0]
            abs_distances = torch.abs(candidates - median.unsqueeze(0))  # [s, P]
            # Indices of β closest distances per coordinate: shape [β, P]
            beta_indices = torch.topk(abs_distances, k=beta, dim=0, largest=False).indices
            # Gather candidate values per coordinate using those indices
            # Transpose to operate per-coordinate: candidates^T is [P, s], indices^T is [P, β]
            gathered = torch.gather(candidates.t(), dim=1, index=beta_indices.t())  # [P, β]
            aggregated_delta = gathered.mean(dim=1)  # [P]

        # Apply to global model
        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = aggregated_delta[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        
        global_model.load_state_dict(global_state)
        return global_model


# class FoolsGoldAggregation(BaseAggregation):
#     """FoolsGold: downweight highly similar client updates (sybil-resistant)."""
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.params = config.get('params', {})
#         self.epsilon = float(self.params.get('epsilon', 1.0e-6))
#         self.topk_ratio = float(self.params.get('topk_ratio', 0.1))
#         self.checkpoints: List[torch.Tensor] = []  # history of normalized updates per round
#
#     def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
#         if not client_results:
#             return global_model
#
#         model_keys = list(client_results[0]['model_state'].keys())
#         global_state = global_model.state_dict()
#         snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
#
#         # Build current round flat deltas
#         updates_list: List[torch.Tensor] = []
#         for cr in client_results:
#             flat = []
#             for k in model_keys:
#                 flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
#             updates_list.append(torch.cat(flat))
#         U = torch.stack(updates_list)  # [n, P]
#
#         # 1) Normalize each client's update (cap norm at 1) and record historical gradients
#         norms = torch.norm(U, p=2, dim=1, keepdim=True)
#         U_normed = U.clone()
#         mask = norms.squeeze(1) > 1
#         if mask.any():
#             U_normed[mask] = U_normed[mask] / norms[mask]
#         # store a CPU numpy snapshot for history accumulation (aligning with FLPoison)
#         self.checkpoints.append(U_normed.detach().cpu())
#         # Sum historical gradients across rounds
#         sumed_updates = torch.stack(self.checkpoints, dim=0).sum(dim=0)  # [n, P]
#
#         # 2) Indicative features mask from last layer of global model (optional)
#         feature_dim = U.size(1)
#         indicative_mask = self._build_indicative_mask_from_model(global_model, feature_dim)  # torch.bool [P]
#
#         # 3) Cosine similarity over indicative features; remove self-similarity
#         X = sumed_updates.to(U.device)
#         X_sel = X[:, indicative_mask]
#         denom = torch.norm(X_sel, p=2, dim=1, keepdim=True).clamp(min=self.epsilon)
#         Xu = X_sel / denom
#         S = Xu @ Xu.t()
#         n = S.size(0)
#         S = S - torch.eye(n, dtype=S.dtype, device=S.device)
#
#         # Pardoning
#         wv = self._pardoning(S)
#
#         # # Weighted aggregate of CURRENT updates U (not historical sum)
#         # wv_tensor = torch.from_numpy(wv.astype('float32')).to(U.device).unsqueeze(1)
#         # gm = (wv_tensor * U).sum(dim=0) if float(wv_tensor.sum().item()) > 0 else U.mean(dim=0)
#         # Weighted aggregate of CURRENT updates U (not historical sum)
#         wv_tensor = torch.from_numpy(wv.astype('float32')).to(U.device).unsqueeze(1)
#
#         # === 修复 Bug：必须对权重进行归一化，否则会造成梯度成倍爆炸 ===
#         weight_sum = float(wv_tensor.sum().item())
#         if weight_sum > 0:
#             wv_tensor = wv_tensor / weight_sum  # 归一化权重使其和为1
#             gm = (wv_tensor * U).sum(dim=0)
#         else:
#             gm = U.mean(dim=0)
#
#         # Unflatten and apply to global model
#         with torch.no_grad():
#             offset = 0
#             for key in model_keys:
#                 numel = snapshot[key].numel()
#                 delta = gm[offset:offset + numel].view_as(snapshot[key])
#                 offset += numel
#                 global_state[key].add_(delta.to(global_state[key].dtype))
#         global_model.load_state_dict(global_state)
#         return global_model
#
#     def _pardoning(self, cos_dist: torch.Tensor):
#         cos = cos_dist.detach().cpu().numpy()
#         max_cs = np.max(cos, axis=1) + self.epsilon
#         for i in range(cos.shape[0]):
#             for j in range(cos.shape[1]):
#                 if i == j:
#                     continue
#                 if max_cs[i] < max_cs[j]:
#                     cos[i, j] *= max_cs[i] / max_cs[j]
#         wv = 1.0 - np.max(cos, axis=1)
#         wv = np.clip(wv, 0.0, 1.0)
#         mx = np.max(wv)
#         if mx > 0:
#             wv = wv / mx
#         wv[wv == 1.0] = 0.99
#         # Logit shaping
#         wv = np.log(wv / (1.0 - wv) + self.epsilon) + 0.5
#         wv[(~np.isfinite(wv)) | (wv > 1.0)] = 1.0
#         wv[wv < 0.0] = 0.0
#         return wv
#
#     def _build_indicative_mask_from_model(self, model: nn.Module, feature_dim: int) -> torch.Tensor:
#         # Try to find the last linear layer weight of shape [C, D]
#         last_w = None
#         for name, param in model.named_parameters():
#             if param.dim() == 2:
#                 last_w = param.detach().float()
#         if last_w is None:
#             return torch.ones(feature_dim, dtype=torch.bool)
#         class_dim, ol_feature_dim = last_w.size(0), last_w.size(1)
#         topk = max(1, int(class_dim * self.topk_ratio))
#         ol_indicative_idx = torch.zeros((class_dim, ol_feature_dim), dtype=torch.bool)
#         # per-class top-k by absolute value
#         vals, idxs = torch.topk(last_w.abs(), k=topk, dim=1)
#         for i in range(class_dim):
#             ol_indicative_idx[i, idxs[i]] = True
#         flat = ol_indicative_idx.view(-1)
#         # Pad at the front to align with end-positioned output layer params
#         if flat.numel() >= feature_dim:
#             return torch.ones(feature_dim, dtype=torch.bool)
#         pad_len = feature_dim - flat.numel()
#         mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), flat], dim=0)
#         return mask

# class FoolsGoldAggregation(BaseAggregation):
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.params = config.get('params', {})
#         self.epsilon = float(self.params.get('epsilon', 1.0e-6))
#         self.topk_ratio = float(self.params.get('topk_ratio', 0.1))
#         self.checkpoints: List[torch.Tensor] = []
#
#     def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
#         if not client_results:
#             return global_model
#
#         model_keys = list(client_results[0]['model_state'].keys())
#         global_state = global_model.state_dict()
#         snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
#
#         updates_list: List[torch.Tensor] = []
#         for cr in client_results:
#             flat = []
#             for k in model_keys:
#                 flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
#             updates_list.append(torch.cat(flat))
#         U = torch.stack(updates_list)
#
#         norms = torch.norm(U, p=2, dim=1, keepdim=True)
#         U_normed = U.clone()
#         mask = norms.squeeze(1) > 1
#         if mask.any():
#             U_normed[mask] = U_normed[mask] / norms[mask]
#
#         self.checkpoints.append(U_normed.detach().cpu())
#         sumed_updates = torch.stack(self.checkpoints, dim=0).sum(dim=0)
#
#         feature_dim = U.size(1)
#         indicative_mask = self._build_indicative_mask_from_model(global_model, feature_dim)
#
#         X = sumed_updates.to(U.device)
#         X_sel = X[:, indicative_mask]
#         denom = torch.norm(X_sel, p=2, dim=1, keepdim=True).clamp(min=self.epsilon)
#         Xu = X_sel / denom
#         S = Xu @ Xu.t()
#         n = S.size(0)
#         S = S - torch.eye(n, dtype=S.dtype, device=S.device)
#
#         wv = self._pardoning(S)
#         wv_tensor = torch.from_numpy(wv.astype('float32')).to(U.device).unsqueeze(1)
#
#         weight_sum = float(wv_tensor.sum().item())
#         if weight_sum > 0:
#             wv_tensor = wv_tensor / weight_sum
#             gm = (wv_tensor * U).sum(dim=0)
#         else:
#             gm = U.mean(dim=0)
#
#         # ======= 【新增代码】将权重分配大于 1e-4 的视为“被接受” =======
#         wv_list = wv_tensor.squeeze().tolist()
#         if not isinstance(wv_list, list): wv_list = [wv_list]
#         self.last_accepted_clients = [
#             client_results[i].get('client_id', -1)
#             for i, w in enumerate(wv_list) if w > 1e-4
#         ]
#         # =============================================================
#
#         with torch.no_grad():
#             offset = 0
#             for key in model_keys:
#                 numel = snapshot[key].numel()
#                 delta = gm[offset:offset + numel].view_as(snapshot[key])
#                 offset += numel
#                 global_state[key].add_(delta.to(global_state[key].dtype))
#         global_model.load_state_dict(global_state)
#         return global_model
#
#     def _pardoning(self, cos_dist: torch.Tensor):
#         import numpy as np
#         cos = cos_dist.detach().cpu().numpy()
#         max_cs = np.max(cos, axis=1) + self.epsilon
#         for i in range(cos.shape[0]):
#             for j in range(cos.shape[1]):
#                 if i == j: continue
#                 if max_cs[i] < max_cs[j]:
#                     cos[i, j] *= max_cs[i] / max_cs[j]
#         wv = 1.0 - np.max(cos, axis=1)
#         wv = np.clip(wv, 0.0, 1.0)
#         mx = np.max(wv)
#         if mx > 0: wv = wv / mx
#         wv[wv == 1.0] = 0.99
#         wv = np.log(wv / (1.0 - wv) + self.epsilon) + 0.5
#         wv[(~np.isfinite(wv)) | (wv > 1.0)] = 1.0
#         wv[wv < 0.0] = 0.0
#         return wv
#
#     def _build_indicative_mask_from_model(self, model: nn.Module, feature_dim: int) -> torch.Tensor:
#         last_w = None
#         for name, param in model.named_parameters():
#             if param.dim() == 2:
#                 last_w = param.detach().float()
#         if last_w is None:
#             return torch.ones(feature_dim, dtype=torch.bool)
#         class_dim, ol_feature_dim = last_w.size(0), last_w.size(1)
#         topk = max(1, int(class_dim * self.topk_ratio))
#         ol_indicative_idx = torch.zeros((class_dim, ol_feature_dim), dtype=torch.bool)
#         vals, idxs = torch.topk(last_w.abs(), k=topk, dim=1)
#         for i in range(class_dim):
#             ol_indicative_idx[i, idxs[i]] = True
#         flat = ol_indicative_idx.view(-1)
#         if flat.numel() >= feature_dim:
#             return torch.ones(feature_dim, dtype=torch.bool)
#         pad_len = feature_dim - flat.numel()
#         mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), flat], dim=0)
#         return mask

class FoolsGoldAggregation(BaseAggregation):
    """FoolsGold: downweight highly similar client updates (sybil-resistant)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.epsilon = float(self.params.get('epsilon', 1.0e-6))
        self.topk_ratio = float(self.params.get('topk_ratio', 0.1))
        self.checkpoints: List[torch.Tensor] = []  # history of normalized updates per round
        # 新增：用于 Server 层统计 MAR / BAR 的监控名单
        self.last_accepted_clients = []

    def aggregate(self, global_model: nn.Module, client_results: List[Dict], round_num: int) -> nn.Module:
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Build current round flat deltas
        updates_list: List[torch.Tensor] = []
        for cr in client_results:
            flat = []
            for k in model_keys:
                flat.append((cr['model_state'][k].detach().float() - snapshot[k]).view(-1))
            updates_list.append(torch.cat(flat))
        U = torch.stack(updates_list)  # [n, P]

        # 1) Normalize each client's update (cap norm at 1) and record historical gradients
        norms = torch.norm(U, p=2, dim=1, keepdim=True)
        U_normed = U.clone()
        mask = norms.squeeze(1) > 1
        if mask.any():
            U_normed[mask] = U_normed[mask] / norms[mask]
        # store a CPU numpy snapshot for history accumulation (aligning with FLPoison)
        self.checkpoints.append(U_normed.detach().cpu())
        # Sum historical gradients across rounds
        sumed_updates = torch.stack(self.checkpoints, dim=0).sum(dim=0)  # [n, P]

        # 2) Indicative features mask from last layer of global model (optional)
        feature_dim = U.size(1)
        indicative_mask = self._build_indicative_mask_from_model(global_model, feature_dim)  # torch.bool [P]

        # 3) Cosine similarity over indicative features; remove self-similarity
        X = sumed_updates.to(U.device)
        X_sel = X[:, indicative_mask]
        denom = torch.norm(X_sel, p=2, dim=1, keepdim=True).clamp(min=self.epsilon)
        Xu = X_sel / denom
        S = Xu @ Xu.t()
        n = S.size(0)
        S = S - torch.eye(n, dtype=S.dtype, device=S.device)

        # Pardoning
        wv = self._pardoning(S)

        # Weighted aggregate of CURRENT updates U (not historical sum)
        wv_tensor = torch.from_numpy(wv.astype('float32')).to(U.device).unsqueeze(1)

        # ==========================================
        # 🟢 【监控层逻辑修改】：动态相对排名惩罚（倒数前五）
        # ==========================================
        # 获取各客户端 ID
        client_ids = [result.get('client_id', i) for i, result in enumerate(client_results)]

        # 将客户端 ID 与其对应的权重打包
        wv_list = wv.tolist()
        client_weights_pair = [(client_ids[i], wv_list[i]) for i in range(len(client_ids))]

        # 按权重从小到大排序
        client_weights_sorted = sorted(client_weights_pair, key=lambda x: x[1])

        # 找出权重最小的 5 个客户端（使用 min 防御总客户端不足 5 的情况）
        num_to_reject = min(5, len(client_weights_sorted))
        rejected_clients = [cid for cid, _ in client_weights_sorted[:num_to_reject]]

        # 更新监控名单（排除权重倒数前五名），专供 Server 统计 MAR/BAR
        self.last_accepted_clients = [
            cid for cid, _ in client_weights_pair if cid not in rejected_clients
        ]
        # ==========================================

        # ==========================================
        # 🔴 【物理聚合层逻辑】：保持原版代码不被破坏
        # ==========================================
        # === 修复 Bug：必须对权重进行归一化，否则会造成梯度成倍爆炸 ===
        weight_sum = float(wv_tensor.sum().item())
        if weight_sum > 0:
            wv_tensor = wv_tensor / weight_sum  # 归一化权重使其和为1
            gm = (wv_tensor * U).sum(dim=0)
        else:
            gm = U.mean(dim=0)

        # Unflatten and apply to global model
        with torch.no_grad():
            offset = 0
            for key in model_keys:
                numel = snapshot[key].numel()
                delta = gm[offset:offset + numel].view_as(snapshot[key])
                offset += numel
                global_state[key].add_(delta.to(global_state[key].dtype))
        global_model.load_state_dict(global_state)
        return global_model

    def _pardoning(self, cos_dist: torch.Tensor):
        cos = cos_dist.detach().cpu().numpy()
        max_cs = np.max(cos, axis=1) + self.epsilon
        for i in range(cos.shape[0]):
            for j in range(cos.shape[1]):
                if i == j:
                    continue
                if max_cs[i] < max_cs[j]:
                    cos[i, j] *= max_cs[i] / max_cs[j]
        wv = 1.0 - np.max(cos, axis=1)
        wv = np.clip(wv, 0.0, 1.0)
        mx = np.max(wv)
        if mx > 0:
            wv = wv / mx
        wv[wv == 1.0] = 0.99
        # Logit shaping
        wv = np.log(wv / (1.0 - wv) + self.epsilon) + 0.5
        wv[(~np.isfinite(wv)) | (wv > 1.0)] = 1.0
        wv[wv < 0.0] = 0.0
        return wv

    def _build_indicative_mask_from_model(self, model: nn.Module, feature_dim: int) -> torch.Tensor:
        # Try to find the last linear layer weight of shape [C, D]
        last_w = None
        for name, param in model.named_parameters():
            if param.dim() == 2:
                last_w = param.detach().float()
        if last_w is None:
            return torch.ones(feature_dim, dtype=torch.bool)
        class_dim, ol_feature_dim = last_w.size(0), last_w.size(1)
        topk = max(1, int(class_dim * self.topk_ratio))
        ol_indicative_idx = torch.zeros((class_dim, ol_feature_dim), dtype=torch.bool)
        # per-class top-k by absolute value
        vals, idxs = torch.topk(last_w.abs(), k=topk, dim=1)
        for i in range(class_dim):
            ol_indicative_idx[i, idxs[i]] = True
        flat = ol_indicative_idx.view(-1)
        # Pad at the front to align with end-positioned output layer params
        if flat.numel() >= feature_dim:
            return torch.ones(feature_dim, dtype=torch.bool)
        pad_len = feature_dim - flat.numel()
        mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), flat], dim=0)
        return mask

class RLRAggregation(BaseAggregation):
    """RLR (Robust Learning Rate): Sign-based aggregation with adaptive learning rates.
    
    RLR computes the sign of each client's update and uses the agreement of signs
    to determine a robust learning rate per parameter. If the majority of clients
    agree on the sign of an update direction, a positive learning rate is applied;
    otherwise, a negative learning rate is used to counteract potential attacks.
    
    Steps per round:
      1) Compute each client's update (delta = local - global).
      2) Compute the sign of each update.
      3) Sum signs to get agreement score per parameter.
      4) If agreement >= threshold: use positive server_lr, else negative server_lr.
      5) Aggregate updates using simple average (equal weights).
      6) Apply aggregated update * robust_lr_vector to global model.
    
    Config keys (under params):
      - server_lr: float (default 1.0) - base learning rate
      - robustLR_threshold: float (default: number of clients / 2) - minimum agreement threshold
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.server_lr = float(self.params.get('server_lr', 1.0))
        self.robustLR_threshold = self.params.get('robustLR_threshold', None)  # Will be set in aggregate if None
        print(f"🔍 RLRAggregation: server_lr: {self.server_lr}, robustLR_threshold: {self.robustLR_threshold}")
    
    def _stack_client_updates(self, model_keys: List[str], global_snapshot: Dict[str, torch.Tensor], 
                              client_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stack client updates into a flat tensor."""
        flat_list = []
        for key in model_keys:
            # Skip batch norm tracking parameters
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            delta = client_state[key].detach().float() - global_snapshot[key]
            flat_list.append(delta.view(-1))
        return torch.cat(flat_list) if flat_list else torch.tensor([])
    
    def _unstack_update(self, flat_update: torch.Tensor, model_keys: List[str], 
                       global_snapshot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unstack flat tensor back into model update dictionary."""
        update_dict = {}
        offset = 0
        for key in model_keys:
            if key.split('.')[-1] == 'num_batches_tracked':
                # Skip batch norm tracking parameters, keep original value
                continue
            numel = global_snapshot[key].numel()
            update_dict[key] = flat_update[offset:offset + numel].view_as(global_snapshot[key])
            offset += numel
        return update_dict
    
    def _compute_robust_lr(self, agent_updates_dict: Dict[int, torch.Tensor], 
                          threshold: float, device: torch.device) -> torch.Tensor:
        """Compute robust learning rate vector based on sign agreement.
        
        This follows the exact logic from RLRorigin.py:
        1. Compute sign of each update
        2. Sum signs and take absolute value to get agreement score
        3. If agreement >= threshold: use positive server_lr, else negative server_lr
        
        Args:
            agent_updates_dict: Dictionary mapping client index to flat update tensor
            threshold: Agreement threshold (minimum number of agreeing signs)
            device: Device to place the result tensor
            
        Returns:
            Learning rate vector of shape [num_params] with values in {server_lr, -server_lr}
        """
        if not agent_updates_dict:
            return torch.tensor([self.server_lr], device=device)
        
        # Get sign of each update (exactly as in original RLR)
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        
        # Sum of signs gives agreement score (absolute value)
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        # Apply threshold exactly as in original: modify in place
        # If agreement < threshold: use negative LR, else use positive LR
        lr_vector = sm_of_signs.clone()
        lr_vector[sm_of_signs < threshold] = -self.server_lr
        lr_vector[sm_of_signs >= threshold] = self.server_lr
        
        return lr_vector.to(device)
    
    def _agg_avg(self, agent_updates_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Simple average aggregation of client updates (equal weights)."""
        if not agent_updates_dict:
            raise ValueError("agent_updates_dict is empty")
        
        sm_updates = sum(agent_updates_dict.values())
        total_data = len(agent_updates_dict)
        return sm_updates / total_data
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using Robust Learning Rate method."""
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        
        # Filter out batch norm tracking parameters from model_keys for update computation
        update_keys = [k for k in model_keys if k.split('.')[-1] != 'num_batches_tracked']
        
        # Build agent updates dictionary: client_idx -> flat update tensor
        agent_updates_dict: Dict[int, torch.Tensor] = {}
        for idx, result in enumerate(client_results):
            client_update = self._stack_client_updates(update_keys, snapshot, result['model_state'])
            if client_update.numel() > 0:
                agent_updates_dict[idx] = client_update
        
        if not agent_updates_dict:
            return global_model
        
        # Determine threshold: default to half the number of clients
        num_clients = len(agent_updates_dict)
        threshold = self.robustLR_threshold
        if threshold is None:
            threshold = float(num_clients / 2)
        threshold = float(threshold)
        
        print(f"🔍 RLRAggregation: num_clients: {num_clients}, threshold: {threshold}")
        
        # Get device from first update
        device = list(agent_updates_dict.values())[0].device
        
        # Compute robust learning rate vector
        lr_vector = self._compute_robust_lr(agent_updates_dict, threshold, device)
        
        # Aggregate updates using simple average
        aggregated_updates = self._agg_avg(agent_updates_dict)
        
        # Apply robust LR to aggregated updates
        robust_updates = lr_vector * aggregated_updates
        
        # Unstack and apply to global model
        with torch.no_grad():
            robust_updates_dict = self._unstack_update(robust_updates, update_keys, snapshot)
            
            for key in update_keys:
                if key in robust_updates_dict:
                    # Ensure dtype compatibility
                    update = robust_updates_dict[key].to(global_state[key].dtype)
                    global_state[key].add_(update)
            
            # Keep batch norm tracking parameters from first client (standard practice)
            for key in model_keys:
                if key.split('.')[-1] == 'num_batches_tracked':
                    if key in client_results[0]['model_state']:
                        global_state[key] = client_results[0]['model_state'][key]
        
        global_model.load_state_dict(global_state)
        return global_model

class MultiMetricAggregation(BaseAggregation):
    """MultiMetric: Multi-dimensional anomaly detection using Mahalanobis distance.
    
    MultiMetric computes three distance metrics between client models:
    1. Cosine distance: (1 - cosine_similarity)^2
    2. Manhattan distance: L1 norm
    3. Length distance: |norm(model_i) - norm(model_j)|
    
    For each client, it sums distances to all other clients for each metric,
    creating a 3D feature vector. It then uses Mahalanobis distance to detect
    outliers and selects the top-k clients (by default 30%) with the lowest
    Mahalanobis scores for aggregation.
    
    Steps per round:
      1) Vectorize each client's full model state.
      2) Compute pairwise cosine, Manhattan, and length distances.
      3) Sum distances for each client to get 3D feature vector.
      4) Compute covariance matrix and Mahalanobis distance for each client.
      5) Select top-k clients (proportion p) with lowest Mahalanobis scores.
      6) Aggregate selected clients using weighted average.
    
    Config keys (under params):
      - proportion: float (default 0.3) - proportion of clients to select (0.3 = 30%)
      - epsilon: float (default 1e-6) - small value for numerical stability
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.proportion = float(self.params.get('proportion', 0.3))
        self.epsilon = float(self.params.get('epsilon', 1e-6))
        print(f"🔍 MultiMetricAggregation: proportion: {self.proportion}, epsilon: {self.epsilon}")
    
    def _vectorize_model(self, model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Vectorize model state into a flat tensor, excluding batch norm tracking."""
        vec = []
        for key, param in model_state.items():
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            vec.append(param.detach().float().view(-1))
        return torch.cat(vec) if vec else torch.tensor([])
    
    def _compute_distances(self, vectorized_models: List[torch.Tensor]) -> tuple:
        """Compute cosine, Manhattan, and length distances for each client.
        
        This follows the exact logic from multimetric.py exec_cuda() method:
        - Cosine distance: 1 - cosine_similarity (not squared, matching exec_cuda)
        - Manhattan distance: L1 norm
        - Length distance: |norm(g_i) - norm(g_j)|
        
        Args:
            vectorized_models: List of flattened model vectors
            
        Returns:
            Tuple of (cos_dis, manhattan_dis, length_dis) lists
        """
        numC = len(vectorized_models)
        cos_dis = [0.0] * numC
        manhattan_dis = [0.0] * numC
        length_dis = [0.0] * numC
        
        # Use torch operations for distance computation (matching exec_cuda)
        # All models should be on the same device (ensured in aggregate method)
        cos_fn = torch.nn.CosineSimilarity(dim=0, eps=self.epsilon)
        
        for i, g_i in enumerate(vectorized_models):
            for j in range(len(vectorized_models)):
                if i != j:
                    g_j = vectorized_models[j]
                    
                    # Cosine distance: 1 - cosine_similarity (exactly as in exec_cuda)
                    cosine_distance = (1 - cos_fn(g_i, g_j)).item()
                    
                    # Manhattan distance: L1 norm (exactly as in exec_cuda)
                    manhattan_distance = torch.norm(g_i - g_j, p=1).item()
                    
                    # Length distance: |norm(g_i) - norm(g_j)| (exactly as in exec_cuda)
                    length_distance = torch.abs(torch.norm(g_i) - torch.norm(g_j)).item()
                    
                    cos_dis[i] += cosine_distance
                    manhattan_dis[i] += manhattan_distance
                    length_dis[i] += length_distance
        
        return cos_dis, manhattan_dis, length_dis
    
    def _compute_mahalanobis_scores(self, cos_dis: List[float], 
                                    manhattan_dis: List[float], 
                                    length_dis: List[float]) -> np.ndarray:
        """Compute Mahalanobis distance scores for each client.
        
        Args:
            cos_dis: List of cosine distance sums
            manhattan_dis: List of Manhattan distance sums
            length_dis: List of length distance sums
            
        Returns:
            Array of Mahalanobis distance scores
        """
        # Stack distances into a matrix: [num_clients, 3]
        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T
        
        # Compute covariance matrix
        cov_matrix = np.cov(tri_distance.T)
        
        # Handle singular matrix by adding small regularization
        try:
            inv_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # If singular, add small identity matrix for regularization
            reg = np.eye(cov_matrix.shape[0]) * self.epsilon
            inv_matrix = np.linalg.inv(cov_matrix + reg)
        
        # Compute Mahalanobis distance for each client
        ma_distances = []
        for i in range(len(cos_dis)):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)
        
        return np.array(ma_distances)
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using MultiMetric method."""
        if not client_results:
            return global_model
        
        # Vectorize all client models and ensure they're on the same device
        vectorized_models = []
        # Get device from global model
        device = next(global_model.parameters()).device
        
        for result in client_results:
            model_vec = self._vectorize_model(result['model_state'])
            if model_vec.numel() > 0:
                # Ensure model vector is on the same device as global model
                model_vec = model_vec.to(device)
                vectorized_models.append(model_vec)
            else:
                # If empty, skip this client
                continue
        
        if len(vectorized_models) == 0:
            return global_model
        
        if len(vectorized_models) == 1:
            # Only one client, use it directly
            selected_indices = [0]
        else:
            # Compute distances
            cos_dis, manhattan_dis, length_dis = self._compute_distances(vectorized_models)
            
            print(f"🔍 MultiMetricAggregation: cos_dis: {cos_dis}")
            print(f"🔍 MultiMetricAggregation: manhattan_dis: {manhattan_dis}")
            print(f"🔍 MultiMetricAggregation: length_dis: {length_dis}")
            
            # Compute Mahalanobis scores
            scores = self._compute_mahalanobis_scores(cos_dis, manhattan_dis, length_dis)
            print(f"🔍 MultiMetricAggregation: Mahalanobis scores: {scores}")
            print(f"🔍 MultiMetricAggregation: sorted indices: {np.argsort(scores)}")
            
            # Select top-k clients with lowest scores (proportion p)
            p = max(0.1, min(self.proportion, 0.9))  # Clamp between 0.1 and 0.9
            p_num = int(p * len(scores))
            p_num = max(1, p_num)  # Ensure at least 1 client is selected
            
            # Get indices of clients with lowest scores
            topk_ind = np.argpartition(scores, p_num)[:p_num]
            selected_indices = topk_ind.tolist()
            print(f"🔍 MultiMetricAggregation: selected_indices: {selected_indices}")
        
        # Filter client_results to match vectorized_models indices
        # (in case some were skipped due to empty vectors)
        valid_results = []
        valid_indices = []
        vec_idx = 0
        for orig_idx, result in enumerate(client_results):
            model_vec = self._vectorize_model(result['model_state'])
            if model_vec.numel() > 0:
                if vec_idx in selected_indices:
                    valid_results.append(result)
                    valid_indices.append(orig_idx)
                vec_idx += 1
        
        if not valid_results:
            # Fallback: use all clients if selection failed
            valid_results = client_results
            print("🔍 MultiMetricAggregation: Warning - no clients selected, using all clients")
        
        # Aggregate selected clients using weighted average
        total_samples = sum(result['samples'] for result in valid_results)
        model_keys = list(valid_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        
        # Prepare aggregated state
        aggregated_state = {}
        for key in model_keys:
            aggregated_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        
        # Weighted aggregation
        with torch.no_grad():
            for result in valid_results:
                weight = result['samples'] / total_samples
                local_state = result['model_state']
                
                for key in model_keys:
                    local_tensor = local_state[key].float()
                    aggregated_state[key] += weight * local_tensor
            
            # Apply to global model (cast to original dtype)
            for key in model_keys:
                if global_state[key].dtype != aggregated_state[key].dtype:
                    aggregated_state[key] = aggregated_state[key].to(global_state[key].dtype)
                global_state[key] = aggregated_state[key]
            
            # Keep batch norm tracking parameters from first selected client
            for key in model_keys:
                if key.split('.')[-1] == 'num_batches_tracked':
                    if key in valid_results[0]['model_state']:
                        global_state[key] = valid_results[0]['model_state'][key]
        
        global_model.load_state_dict(global_state)
        return global_model

# class DnCAggregation(BaseAggregation):
#     """DnC (Divide and Conquer): Robust aggregation using iterative SVD-based filtering.
#
#     DnC uses an iterative approach to filter out Byzantine clients:
#     1. For each iteration:
#        - Randomly samples a subset of dimensions from client updates
#        - Computes the mean and centers the updates
#        - Uses SVD to find the principal component (first right singular vector)
#        - Scores each client by squared projection on the principal component
#        - Selects clients with lowest scores (filters out filter_frac * num_byzantine)
#     2. Takes the intersection of selected clients across all iterations
#     3. Aggregates only the selected benign clients
#
#     This method is from the paper "Manipulating the Byzantine: Optimizing
#     Model Poisoning Attacks and Defenses for Federated Learning"
#
#     Steps per round:
#       1) Compute client updates (deltas = local - global).
#       2) For num_iters iterations:
#          - Sample random subset of dimensions (sub_dim)
#          - Compute principal component via SVD
#          - Score clients and filter outliers
#       3) Take intersection of selected clients across iterations.
#       4) Aggregate selected clients using weighted average.
#
#     Config keys (under params):
#       - num_byzantine: int (default: None, will be estimated) - number of Byzantine clients
#       - sub_dim: int (default 10000) - number of dimensions to sample per iteration
#       - num_iters: int (default 5) - number of filtering iterations
#       - filter_frac: float (default 1.0) - fraction of byzantine clients to filter per iteration
#       - byzantine_ratio: float (default 0.1) - ratio to estimate num_byzantine if not provided
#     """
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.params = config.get('params', {})
#         self.num_byzantine = self.params.get('num_byzantine', None)
#         self.sub_dim = int(self.params.get('sub_dim', 10000))
#         self.num_iters = int(self.params.get('num_iters', 5))
#         self.filter_frac = float(self.params.get('filter_frac', 1.0))
#         self.byzantine_ratio = float(self.params.get('byzantine_ratio', 0.1))
#         print(f"🔍 DnCAggregation: num_iters: {self.num_iters}, sub_dim: {self.sub_dim}, filter_frac: {self.filter_frac}")
#
#     def _vectorize_updates(self, model_keys: List[str], global_snapshot: Dict[str, torch.Tensor],
#                           client_states: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
#         """Vectorize client updates into flat tensors.
#
#         Args:
#             model_keys: List of model parameter keys
#             global_snapshot: Snapshot of global model state
#             client_states: List of client model states
#
#         Returns:
#             List of flattened update tensors
#         """
#         updates = []
#         for client_state in client_states:
#             vec = []
#             for key in model_keys:
#                 # Skip batch norm tracking and running stats (matching original)
#                 if (key.split('.')[-1] == 'num_batches_tracked' or
#                     key.split('.')[-1] == 'running_mean' or
#                     key.split('.')[-1] == 'running_var'):
#                     continue
#                 delta = client_state[key].detach().float() - global_snapshot[key]
#                 vec.append(delta.view(-1))
#             if vec:
#                 updates.append(torch.cat(vec))
#             else:
#                 # If no valid parameters, create empty tensor
#                 updates.append(torch.tensor([]))
#         return updates
#
#     def _select_benign_clients_iteration(self, updates: List[torch.Tensor],
#                                          num_byzantine: int,
#                                          device: torch.device) -> List[int]:
#         """Select benign clients for one iteration using SVD-based filtering.
#
#         Args:
#             updates: List of flattened update tensors
#             num_byzantine: Estimated number of Byzantine clients
#             device: Device to perform computations on
#
#         Returns:
#             List of selected client indices
#         """
#         if len(updates) == 0:
#             return []
#
#         # Filter out empty updates
#         valid_updates = []
#         valid_indices = []
#         for idx, upd in enumerate(updates):
#             if upd.numel() > 0:
#                 valid_updates.append(upd.to(device))
#                 valid_indices.append(idx)
#
#         if len(valid_updates) == 0:
#             return []
#
#         d = len(valid_updates[0])
#         if d == 0:
#             return list(range(len(updates)))
#
#         # Sample random subset of dimensions
#         sub_dim = min(self.sub_dim, d)
#         indices = torch.randperm(d, device=device)[:sub_dim]
#
#         # Extract subset of updates
#         sub_updates = torch.stack([upd[indices] for upd in valid_updates])
#
#         # Compute mean and center updates
#         mu = sub_updates.mean(dim=0)
#         centered_updates = sub_updates - mu
#
#         # Compute SVD to get principal component
#         # SVD on centered_updates (shape [num_clients, sub_dim]) returns U, S, V^T
#         # We want the first right singular vector, which is V^T[0, :] (first row of V^T)
#         try:
#             U, S, Vt = torch.linalg.svd(centered_updates, full_matrices=False)
#             # First right singular vector is the first row of V^T (matching original)
#             v = Vt[0, :]  # Principal component
#         except Exception as e:
#             # If SVD fails, return all clients
#             print(f"🔍 DnCAggregation: SVD failed: {e}, returning all clients")
#             return list(range(len(updates)))
#
#         # Compute scores: squared dot product with principal component
#         # Match original: for each update in sub_updates, compute (update - mu) dot v
#         scores = []
#         for update in sub_updates:
#             centered = update - mu
#             score = (torch.dot(centered, v) ** 2).item()
#             scores.append(score)
#
#         scores = np.array(scores)
#
#         # Select clients with lowest scores (filter out filter_frac * num_byzantine)
#         num_to_filter = int(self.filter_frac * num_byzantine)
#         num_to_select = max(1, len(valid_updates) - num_to_filter)
#         good_indices = scores.argsort()[:num_to_select]
#
#         # Map back to original indices
#         selected_indices = [valid_indices[idx] for idx in good_indices]
#
#         return selected_indices
#
#     def aggregate(self, global_model: nn.Module, client_results: List[Dict],
#                   round_num: int) -> nn.Module:
#         """Aggregate client updates using DnC (Divide and Conquer) method."""
#         if not client_results:
#             return global_model
#
#         model_keys = list(client_results[0]['model_state'].keys())
#         global_state = global_model.state_dict()
#         snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
#
#         # Get device from global model
#         device = next(global_model.parameters()).device
#
#         # Vectorize client updates
#         client_states = [result['model_state'] for result in client_results]
#         updates = self._vectorize_updates(model_keys, snapshot, client_states)
#
#         if len(updates) == 0 or all(upd.numel() == 0 for upd in updates):
#             return global_model
#
#         # Estimate number of Byzantine clients if not provided
#         num_clients = len(client_results)
#         if self.num_byzantine is None:
#             # Estimate based on byzantine_ratio and number of clients
#             # Original uses: args.frac * args.num_users * 0.1
#             # Since we don't have frac here, we use byzantine_ratio directly
#             num_byzantine = max(1, int(self.byzantine_ratio * num_clients))
#         else:
#             num_byzantine = int(self.num_byzantine)
#
#         num_byzantine = min(num_byzantine, num_clients - 1)  # Can't filter all clients
#         print(f"🔍 DnCAggregation: num_clients: {num_clients}, estimated num_byzantine: {num_byzantine}")
#
#         # Perform multiple iterations and take intersection
#         benign_ids_list = []
#         for i in range(self.num_iters):
#             selected = self._select_benign_clients_iteration(updates, num_byzantine, device)
#             if selected:
#                 benign_ids_list.append(set(selected))
#             else:
#                 # If selection fails, use all clients for this iteration
#                 benign_ids_list.append(set(range(num_clients)))
#
#         # Take intersection of all iterations
#         if benign_ids_list:
#             intersection_set = benign_ids_list[0].copy()
#             for benign_set in benign_ids_list[1:]:
#                 intersection_set.intersection_update(benign_set)
#             benign_ids = sorted(list(intersection_set))
#         else:
#             # Fallback: use all clients
#             benign_ids = list(range(num_clients))
#
#         if not benign_ids:
#             # If intersection is empty, use all clients
#             print("🔍 DnCAggregation: Warning - intersection is empty, using all clients")
#             benign_ids = list(range(num_clients))
#
#         print(f"🔍 DnCAggregation: selected benign client indices: {benign_ids}")
#
#         # Aggregate selected clients using weighted average
#         selected_results = [client_results[i] for i in benign_ids]
#         total_samples = sum(result['samples'] for result in selected_results)
#
#         if total_samples == 0:
#             # Fallback: use equal weights
#             total_samples = len(selected_results)
#             weights = [1.0 / len(selected_results)] * len(selected_results)
#         else:
#             weights = [result['samples'] / total_samples for result in selected_results]
#
#         # Prepare aggregated state
#         aggregated_state = {}
#         for key in model_keys:
#             aggregated_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
#
#         # Weighted aggregation
#         with torch.no_grad():
#             for result, weight in zip(selected_results, weights):
#                 local_state = result['model_state']
#
#                 for key in model_keys:
#                     local_tensor = local_state[key].float()
#                     aggregated_state[key] += weight * local_tensor
#
#             # Apply to global model (cast to original dtype)
#             for key in model_keys:
#                 if global_state[key].dtype != aggregated_state[key].dtype:
#                     aggregated_state[key] = aggregated_state[key].to(global_state[key].dtype)
#                 global_state[key] = aggregated_state[key]
#
#             # Keep batch norm tracking parameters from first selected client
#             for key in model_keys:
#                 if key.split('.')[-1] == 'num_batches_tracked':
#                     if key in selected_results[0]['model_state']:
#                         global_state[key] = selected_results[0]['model_state'][key]
#
#         global_model.load_state_dict(global_state)
#         return global_model

class DnCAggregation(BaseAggregation):
    """DnC (Divide and Conquer): Robust aggregation using iterative SVD-based filtering.

    DnC uses an iterative approach to filter out Byzantine clients:
    1. For each iteration:
       - Randomly samples a subset of dimensions from client updates
       - Computes the mean and centers the updates
       - Uses SVD to find the principal component (first right singular vector)
       - Scores each client by squared projection on the principal component
       - Selects clients with lowest scores (filters out filter_frac * num_byzantine)
    2. Takes the intersection of selected clients across all iterations
    3. Aggregates only the selected benign clients

    This method is from the paper "Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning"

    Steps per round:
      1) Compute client updates (deltas = local - global).
      2) For num_iters iterations:
         - Sample random subset of dimensions (sub_dim)
         - Compute principal component via SVD
         - Score clients and filter outliers
      3) Take intersection of selected clients across iterations.
      4) Aggregate selected clients using weighted average.

    Config keys (under params):
      - num_byzantine: int (default: None, will be estimated) - number of Byzantine clients
      - sub_dim: int (default 10000) - number of dimensions to sample per iteration
      - num_iters: int (default 5) - number of filtering iterations
      - filter_frac: float (default 1.0) - fraction of byzantine clients to filter per iteration
      - byzantine_ratio: float (default 0.1) - ratio to estimate num_byzantine if not provided
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.num_byzantine = self.params.get('num_byzantine', None)
        self.sub_dim = int(self.params.get('sub_dim', 10000))
        self.num_iters = int(self.params.get('num_iters', 5))
        self.filter_frac = float(self.params.get('filter_frac', 1.0))
        self.byzantine_ratio = float(self.params.get('byzantine_ratio', 0.1))
        print(
            f"🔍 DnCAggregation: num_iters: {self.num_iters}, sub_dim: {self.sub_dim}, filter_frac: {self.filter_frac}")

    def _vectorize_updates(self, model_keys: List[str], global_snapshot: Dict[str, torch.Tensor],
                           client_states: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """Vectorize client updates into flat tensors.

        Args:
            model_keys: List of model parameter keys
            global_snapshot: Snapshot of global model state
            client_states: List of client model states

        Returns:
            List of flattened update tensors
        """
        updates = []
        for client_state in client_states:
            vec = []
            for key in model_keys:
                # Skip batch norm tracking and running stats (matching original)
                if (key.split('.')[-1] == 'num_batches_tracked' or
                        key.split('.')[-1] == 'running_mean' or
                        key.split('.')[-1] == 'running_var'):
                    continue
                delta = client_state[key].detach().float() - global_snapshot[key]
                vec.append(delta.view(-1))
            if vec:
                updates.append(torch.cat(vec))
            else:
                # If no valid parameters, create empty tensor
                updates.append(torch.tensor([]))
        return updates

    def _select_benign_clients_iteration(self, updates: List[torch.Tensor],
                                         num_byzantine: int,
                                         device: torch.device) -> List[int]:
        """Select benign clients for one iteration using SVD-based filtering.

        Args:
            updates: List of flattened update tensors
            num_byzantine: Estimated number of Byzantine clients
            device: Device to perform computations on

        Returns:
            List of selected client indices
        """
        if len(updates) == 0:
            return []

        # Filter out empty updates
        valid_updates = []
        valid_indices = []
        for idx, upd in enumerate(updates):
            if upd.numel() > 0:
                valid_updates.append(upd.to(device))
                valid_indices.append(idx)

        if len(valid_updates) == 0:
            return []

        d = len(valid_updates[0])
        if d == 0:
            return list(range(len(updates)))

        # Sample random subset of dimensions
        sub_dim = min(self.sub_dim, d)
        indices = torch.randperm(d, device=device)[:sub_dim]

        # Extract subset of updates
        sub_updates = torch.stack([upd[indices] for upd in valid_updates])

        # Compute mean and center updates
        mu = sub_updates.mean(dim=0)
        centered_updates = sub_updates - mu

        # Compute SVD to get principal component
        # SVD on centered_updates (shape [num_clients, sub_dim]) returns U, S, V^T
        # We want the first right singular vector, which is V^T[0, :] (first row of V^T)
        try:
            U, S, Vt = torch.linalg.svd(centered_updates, full_matrices=False)
            # First right singular vector is the first row of V^T (matching original)
            v = Vt[0, :]  # Principal component
        except Exception as e:
            # If SVD fails, return all clients
            print(f"🔍 DnCAggregation: SVD failed: {e}, returning all clients")
            return list(range(len(updates)))

        # Compute scores: squared dot product with principal component
        # Match original: for each update in sub_updates, compute (update - mu) dot v
        scores = []
        for update in sub_updates:
            centered = update - mu
            score = (torch.dot(centered, v) ** 2).item()
            scores.append(score)

        scores = np.array(scores)

        # Select clients with lowest scores (filter out filter_frac * num_byzantine)
        num_to_filter = int(self.filter_frac * num_byzantine)
        num_to_select = max(1, len(valid_updates) - num_to_filter)
        good_indices = scores.argsort()[:num_to_select]

        # Map back to original indices
        selected_indices = [valid_indices[idx] for idx in good_indices]

        return selected_indices

    def aggregate(self, global_model: nn.Module, client_results: List[Dict],
                  round_num: int) -> nn.Module:
        """Aggregate client updates using DnC (Divide and Conquer) method."""
        if not client_results:
            return global_model

        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}

        # Get device from global model
        device = next(global_model.parameters()).device

        # Vectorize client updates
        client_states = [result['model_state'] for result in client_results]
        updates = self._vectorize_updates(model_keys, snapshot, client_states)

        if len(updates) == 0 or all(upd.numel() == 0 for upd in updates):
            return global_model

        # Estimate number of Byzantine clients if not provided
        num_clients = len(client_results)
        if self.num_byzantine is None:
            # Estimate based on byzantine_ratio and number of clients
            # Original uses: args.frac * args.num_users * 0.1
            # Since we don't have frac here, we use byzantine_ratio directly
            num_byzantine = max(1, int(self.byzantine_ratio * num_clients))
        else:
            num_byzantine = int(self.num_byzantine)

        num_byzantine = min(num_byzantine, num_clients - 1)  # Can't filter all clients
        print(f"🔍 DnCAggregation: num_clients: {num_clients}, estimated num_byzantine: {num_byzantine}")

        # Perform multiple iterations and take intersection
        benign_ids_list = []
        for i in range(self.num_iters):
            selected = self._select_benign_clients_iteration(updates, num_byzantine, device)
            if selected:
                benign_ids_list.append(set(selected))
            else:
                # If selection fails, use all clients for this iteration
                benign_ids_list.append(set(range(num_clients)))

        # Take intersection of all iterations
        if benign_ids_list:
            intersection_set = benign_ids_list[0].copy()
            for benign_set in benign_ids_list[1:]:
                intersection_set.intersection_update(benign_set)
            benign_ids = sorted(list(intersection_set))
        else:
            # Fallback: use all clients
            benign_ids = list(range(num_clients))

        if not benign_ids:
            # If intersection is empty, use all clients
            print("🔍 DnCAggregation: Warning - intersection is empty, using all clients")
            benign_ids = list(range(num_clients))

        print(f"🔍 DnCAggregation: selected benign client indices: {benign_ids}")

        # ============ 【新增代码：把被接受的 ID 暴露给 Server】 ============
        self.last_accepted_clients = [client_results[i].get('client_id', -1) for i in benign_ids]
        # ===============================================================

        # Aggregate selected clients using weighted average
        selected_results = [client_results[i] for i in benign_ids]
        total_samples = sum(result['samples'] for result in selected_results)

        if total_samples == 0:
            # Fallback: use equal weights
            total_samples = len(selected_results)
            weights = [1.0 / len(selected_results)] * len(selected_results)
        else:
            weights = [result['samples'] / total_samples for result in selected_results]

        # Prepare aggregated state
        aggregated_state = {}
        for key in model_keys:
            aggregated_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

        # Weighted aggregation
        with torch.no_grad():
            for result, weight in zip(selected_results, weights):
                local_state = result['model_state']

                for key in model_keys:
                    local_tensor = local_state[key].float()
                    aggregated_state[key] += weight * local_tensor

            # Apply to global model (cast to original dtype)
            for key in model_keys:
                if global_state[key].dtype != aggregated_state[key].dtype:
                    aggregated_state[key] = aggregated_state[key].to(global_state[key].dtype)
                global_state[key] = aggregated_state[key]

            # Keep batch norm tracking parameters from first selected client
            for key in model_keys:
                if key.split('.')[-1] == 'num_batches_tracked':
                    if key in selected_results[0]['model_state']:
                        global_state[key] = selected_results[0]['model_state'][key]

        global_model.load_state_dict(global_state)
        return global_model

class FLAREAggregation(BaseAggregation):
    """FLARE: Feature-based defense using Maximum Mean Discrepancy (MMD) and voting.
    
    FLARE extracts penultimate layer representations (PLR) from client models on a
    central dataset, computes MMD distances between clients, and uses a voting
    mechanism to assign trust scores. Clients vote for their k nearest neighbors,
    and trust scores are computed based on votes.
    
    Steps per round:
      1) Extract PLR features from each client model on central dataset.
      2) Compute pairwise MMD distances between client features.
      3) Each client votes for its k nearest neighbors (k = 50% of clients).
      4) Compute trust scores based on vote counts.
      5) Aggregate client updates using trust score-weighted average.
    
    This method requires:
      - A central dataset (can be passed via kwargs['central_dataset'] or config)
      - Models with feature extraction capability (get_feature method or manual extraction)
    
    Config keys (under params):
      - k_ratio: float (default 0.5) - ratio of clients for k-nearest neighbors voting
      - sigma: float (default 1.0) - kernel parameter for MMD computation
      - epsilon: float (default 1e-6) - small value for numerical stability
      - batch_size: int (default 64) - batch size for feature extraction
      - central_dataset_size: int (default None) - size of central dataset if generating from config
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.k_ratio = float(self.params.get('k_ratio', 0.5))
        self.sigma = float(self.params.get('sigma', 1.0))
        self.epsilon = float(self.params.get('epsilon', 1e-6))
        self.batch_size = int(self.params.get('batch_size', 64))
        self.central_dataset_size = self.params.get('central_dataset_size', None)
        print(f"🔍 FLAREAggregation: k_ratio: {self.k_ratio}, sigma: {self.sigma}, batch_size: {self.batch_size}")
    
    def _extract_features(self, model: nn.Module, dataset, device: torch.device) -> torch.Tensor:
        """Extract penultimate layer features from model on dataset.
        
        This method tries to extract features using:
        1. model.get_feature() method if available (preferred)
        2. Manual extraction by hooking into the penultimate layer
        3. Fallback to model output (logits) if penultimate layer cannot be accessed
        
        Args:
            model: Model to extract features from
            dataset: Dataset to extract features on (PyTorch Dataset with __getitem__)
            device: Device to perform computation on
            
        Returns:
            Tensor of features [num_samples, feature_dim]
        """
        model.eval()
        features_list = []
        
        # Check if model has get_feature method (preferred method)
        has_get_feature = hasattr(model, 'get_feature') and callable(getattr(model, 'get_feature'))
        
        # Create data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        if not has_get_feature:
            print("🔍 FLAREAggregation: Warning - model does not have get_feature method. "
                  "Using model output (logits) as features. This may affect FLARE performance.")
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                
                if has_get_feature:
                    # Use model's get_feature method (matching original FLARE)
                    features = model.get_feature(images)
                else:
                    # Fallback: use model output as features
                    # Note: This is not ideal but allows FLARE to run
                    # For best results, models should implement get_feature method
                    try:
                        output = model(images)
                        features = output
                    except Exception as e:
                        print(f"🔍 FLAREAggregation: Error during feature extraction: {e}")
                        # Last resort: use a dummy feature vector
                        features = torch.zeros(images.size(0), 10, device=device)
                
                # Flatten features if needed (matching original FLARE)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                
                features_list.append(features.cpu())
        
        if features_list:
            return torch.cat(features_list, dim=0)
        else:
            # Return empty tensor if no features extracted
            return torch.tensor([])
    
    def _kernel_function(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute RBF kernel between two feature vectors.
        
        Matching original FLARE kernel_function exactly:
        kernel(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        
        Args:
            x: First feature vector
            y: Second feature vector
            
        Returns:
            Kernel value
        """
        diff = x - y
        norm_sq = torch.norm(diff) ** 2
        return float(torch.exp(-norm_sq / (2 * self.sigma ** 2)).item())
    
    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Maximum Mean Discrepancy (MMD) between two feature sets.
        
        Matching original FLARE compute_mmd exactly:
        MMD = sum(xx_kernel) / (m * (m-1)) + sum(yy_kernel) / (n * (n-1)) - 2 * sum(xy_kernel) / (m * n)
        
        Args:
            x: First feature set [m, feature_dim]
            y: Second feature set [n, feature_dim]
            
        Returns:
            MMD value
        """
        m = x.size(0)
        n = y.size(0)
        
        if m == 0 or n == 0:
            return float('inf')
        
        # Compute kernel matrices (matching original FLARE exactly)
        xx_kernel = torch.zeros((m, m))
        yy_kernel = torch.zeros((n, n))
        xy_kernel = torch.zeros((m, n))
        
        for i in range(m):
            for j in range(i, m):
                k_val = self._kernel_function(x[i], x[j])
                xx_kernel[i, j] = k_val
                xx_kernel[j, i] = k_val
        
        for i in range(n):
            for j in range(i, n):
                k_val = self._kernel_function(y[i], y[j])
                yy_kernel[i, j] = k_val
                yy_kernel[j, i] = k_val
        
        for i in range(m):
            for j in range(n):
                xy_kernel[i, j] = self._kernel_function(x[i], y[j])
        
        # Compute MMD statistic (matching original FLARE exactly)
        # Note: Original code assumes m > 1 and n > 1 (would crash otherwise)
        # We handle edge cases gracefully
        if m > 1:
            term1 = torch.sum(xx_kernel) / (m * (m - 1))
        else:
            term1 = torch.tensor(0.0)
        
        if n > 1:
            term2 = torch.sum(yy_kernel) / (n * (n - 1))
        else:
            term2 = torch.tensor(0.0)
        
        term3 = 2 * torch.sum(xy_kernel) / (m * n)
        
        mmd = term1 + term2 - term3
        return float(mmd.item())
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int, **kwargs) -> nn.Module:
        """Aggregate client updates using FLARE method.
        
        Args:
            global_model: Global model
            client_results: List of client results with 'model_state' and 'samples'
            round_num: Current round number
            **kwargs: Additional arguments, can include:
                - central_dataset: PyTorch Dataset to use for feature extraction
                - central_dataset_indices: List of indices to use from a dataset
        
        Returns:
            Updated global model
        """
        if not client_results:
            return global_model
        
        # Get central dataset from kwargs or config
        central_dataset = kwargs.get('central_dataset', None)
        if central_dataset is None:
            # Try to get from config
            central_dataset = self.params.get('central_dataset', None)
        
        if central_dataset is None:
            raise ValueError(
                "FLAREAggregation requires a central_dataset. "
                "Please provide it via kwargs['central_dataset'] or config['params']['central_dataset']"
            )
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        device = next(global_model.parameters()).device
        
        # Extract features from each client model
        print(f"🔍 FLAREAggregation: Extracting features from {len(client_results)} clients")
        w_features = []
        client_models = []
        
        for result in client_results:
            # Create a copy of global model and load client state
            client_model = copy.deepcopy(global_model)
            client_model.load_state_dict(result['model_state'])
            client_models.append(client_model)
            
            # Extract features
            features = self._extract_features(client_model, central_dataset, device)
            w_features.append(features)
        
        if not w_features or any(f.numel() == 0 for f in w_features):
            print("🔍 FLAREAggregation: Warning - could not extract features, falling back to FedAvg")
            # Fallback to FedAvg
            total_samples = sum(result['samples'] for result in client_results)
            aggregated_state = {}
            for key in model_keys:
                aggregated_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            
            with torch.no_grad():
                for result in client_results:
                    weight = result['samples'] / total_samples
                    local_state = result['model_state']
                    for key in model_keys:
                        aggregated_state[key] += weight * local_state[key].float()
                
                for key in model_keys:
                    if global_state[key].dtype != aggregated_state[key].dtype:
                        aggregated_state[key] = aggregated_state[key].to(global_state[key].dtype)
                    global_state[key] = aggregated_state[key]
            
            global_model.load_state_dict(global_state)
            return global_model
        
        # Compute pairwise MMD distances
        # Match original FLARE: for each pair (i,j) with i < j, append d(i,j) to both lists
        num_clients = len(client_results)
        distance_list = [[] for _ in range(num_clients)]
        
        print(f"🔍 FLAREAggregation: Computing MMD distances between {num_clients} clients")
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                mmd_score = self._compute_mmd(w_features[i], w_features[j])
                distance_list[i].append(mmd_score)
                distance_list[j].append(mmd_score)
        
        # distance_list[i] now contains distances to all other clients:
        # - First i elements: distances from clients j < i (d(j,i) for j=0..i-1)
        # - Remaining elements: distances to clients j > i (d(i,j) for j=i+1..n-1)
        # So distance_list[i] = [d(0,i), d(1,i), ..., d(i-1,i), d(i,i+1), d(i,i+2), ..., d(i,n-1)]
        # Index mapping: distance_list[i][idx] corresponds to client:
        #   - If idx < i: client idx
        #   - If idx >= i: client (idx + 1)
        
        # Voting mechanism: each client votes for its k nearest neighbors
        # Match original FLARE voting logic exactly
        k = max(1, round(num_clients * self.k_ratio))
        vote_counter = [0] * num_clients
        
        for i in range(num_clients):
            # Get sorted indices of distances (smallest to largest)
            distances = distance_list[i]
            IDs = np.argsort(distances)
            
            # Vote for first k nearest neighbors
            # Map distance_list index to actual client index (matching original logic)
            for j in range(len(IDs)):
                idx_in_dist_list = int(IDs[j])
                # Map to client ID: if idx >= i, client_id = idx + 1, else client_id = idx
                if idx_in_dist_list >= i:
                    client_id = idx_in_dist_list + 1
                else:
                    client_id = idx_in_dist_list
                
                vote_counter[client_id] += 1
                if j + 1 >= k:  # Vote for first k elements (matching original)
                    break
        
        # Compute trust scores
        total_votes = sum(vote_counter)
        if total_votes == 0:
            # Fallback: equal trust scores
            trust_scores = [1.0 / num_clients] * num_clients
        else:
            trust_scores = [x / total_votes for x in vote_counter]
        
        print(f"🔍 FLAREAggregation: trust_scores: {trust_scores}")
        
        # Aggregate using trust score-weighted average
        # Note: FLARE aggregates updates (deltas), not full models
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        aggregated_update = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in snapshot.items()}
        
        with torch.no_grad():
            for i, result in enumerate(client_results):
                trust_score = trust_scores[i]
                local_state = result['model_state']
                
                for key in model_keys:
                    if key.split('.')[-1] == 'num_batches_tracked':
                        continue
                    delta = local_state[key].detach().float() - snapshot[key]
                    aggregated_update[key] += trust_score * delta
            
            # Apply aggregated update to global model
            for key in model_keys:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                if global_state[key].dtype != aggregated_update[key].dtype:
                    aggregated_update[key] = aggregated_update[key].to(global_state[key].dtype)
                global_state[key].add_(aggregated_update[key])
            
            # Keep batch norm tracking parameters from first client
            for key in model_keys:
                if key.split('.')[-1] == 'num_batches_tracked':
                    if key in client_results[0]['model_state']:
                        global_state[key] = client_results[0]['model_state'][key]
        
        global_model.load_state_dict(global_state)
        return global_model

class LASAAggregation(BaseAggregation):
    """LASA: Layer-wise Adaptive Secure Aggregation.
    
    LASA performs layer-wise filtering of client updates using:
    1. Norm clipping based on median norm
    2. Top-k sparsification of each client's update
    3. Median Z-score (MZ-score) filtering on two metrics:
       - Norm filtering: Filter based on L2 norm of layer updates
       - Sign filtering: Filter based on sign agreement
    4. Aggregate only benign clients (intersection of norm and sign filters)
    
    Steps per round:
      1) Compute client updates (deltas from global model).
      2) Clip updates based on median norm.
      3) Sparsify each client's update (top-k largest values).
      4) For each layer:
         a) Compute L2 norms and filter using MZ-score (norm_bound).
         b) Compute sign agreement and filter using MZ-score (sign_bound).
         c) Take intersection of benign clients.
      5) Aggregate norm-clipped updates from benign clients only.
    
    Config keys (under params):
      - norm_bound: float (default 2.0) - MZ-score bound for norm filtering (1.0 for CIFAR10/100)
      - sign_bound: float (default 1.0) - MZ-score bound for sign filtering
      - sparsity: float (default 0.3) - sparsification ratio (fraction of values to zero out)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.norm_bound = float(self.params.get('norm_bound', 2.0))
        self.sign_bound = float(self.params.get('sign_bound', 1.0))
        self.sparsity = float(self.params.get('sparsity', 0.3))
        print(f"🔍 LASAAggregation: norm_bound: {self.norm_bound}, sign_bound: {self.sign_bound}, sparsity: {self.sparsity}")
    
    def _vectorize_update(self, update_dict: Dict[str, torch.Tensor], 
                         model_keys: List[str]) -> np.ndarray:
        """Vectorize update dictionary into a flat numpy array."""
        vec_list = []
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(update_dict[key].detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])
    
    def _unvectorize_update(self, vector: np.ndarray, model_keys: List[str],
                           global_snapshot: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unvectorize flat numpy array back into update dictionary."""
        update_dict = {}
        offset = 0
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            numel = global_snapshot[key].numel()
            update_dict[key] = torch.from_numpy(
                vector[offset:offset + numel].reshape(global_snapshot[key].shape)
            ).float()
            offset += numel
        return update_dict
    
    def _sparse_update(self, update_dict: Dict[str, torch.Tensor], 
                      model_keys: List[str]) -> Dict[str, torch.Tensor]:
        """Sparsify update using top-k largest strategy.
        
        This function sparsifies convolutional and fully-connected layers
        by keeping only the top-k largest values (k = 1 - sparsity).
        Matching original LASA sparse_update logic exactly.
        
        Args:
            update_dict: Dictionary of layer updates
            model_keys: List of model keys
            
        Returns:
            Sparsified update dictionary (modified in place conceptually, but we return new dict)
        """
        # If sparsity is 0.0, no sparsification needed (matching original: returns mask of ones)
        if self.sparsity == 0.0:
            # Original returns mask of ones, but we just return update_dict unchanged
            return update_dict
        
        # Collect absolute values from conv and fc layers only
        weight_abs_list = []
        mask_dict = {}
        
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            # Only sparsify conv (4D) and fc (2D) layers
            if len(update_dict[key].shape) == 4 or len(update_dict[key].shape) == 2:
                weight_abs = np.abs(update_dict[key].detach().cpu().numpy())
                weight_abs_list.append(weight_abs.flatten())
                mask_dict[key] = np.ones_like(weight_abs, dtype=np.float32)
        
        if not weight_abs_list:
            return update_dict
        
        # Gather all scores and find top-k threshold
        all_scores = np.concatenate(weight_abs_list)
        num_topk = int(len(all_scores) * (1 - self.sparsity))
        
        if num_topk == 0:
            # If sparsity is 1.0, zero out everything
            for key in mask_dict.keys():
                update_dict[key] = torch.zeros_like(update_dict[key])
            return update_dict
        
        # Find k-th largest value
        kth_largest = np.partition(all_scores, -num_topk)[-num_topk]
        
        # Apply mask: set values <= kth_largest to 0
        for key in mask_dict.keys():
            update_np = update_dict[key].detach().cpu().numpy()
            mask = np.abs(update_np) > kth_largest
            masked_update = update_np * mask
            update_dict[key] = torch.from_numpy(masked_update).to(update_dict[key].device).float()
        
        return update_dict
    
    def _mz_score(self, values: np.ndarray, bound: float) -> np.ndarray:
        """Compute Median Z-score and filter values within bound.
        
        Matching original LASA mz_score exactly:
        MZ-score = |value - median| / std
        
        Args:
            values: Array of values to filter (will be modified in place in original, but we don't)
            bound: Threshold bound for MZ-score
            
        Returns:
            Array of indices where MZ-score < bound
        """
        if len(values) == 0:
            return np.array([])
        
        # Make a copy to avoid modifying original (original modifies in place)
        values_copy = values.copy()
        med = np.median(values_copy)
        std = np.std(values_copy)
        
        if std == 0:
            # If std is 0, all values are the same, return all indices
            return np.arange(len(values_copy))
        
        # Compute MZ-score for each value (matching original logic)
        # Original modifies values in place, but we compute scores separately
        mz_scores = np.abs((values_copy - med) / std)
        
        # Return indices where MZ-score < bound
        # Use argwhere and squeeze to match original behavior
        result = np.argwhere(mz_scores < bound)
        if result.size == 0:
            return np.array([])
        return result.squeeze(-1) if result.ndim > 1 else result
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using LASA method."""
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        device = next(global_model.parameters()).device
        
        num_clients = len(client_results)
        
        # Step 1: Compute client updates (deltas)
        dict_form_updates = []
        for result in client_results:
            update_dict = {}
            local_state = result['model_state']
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                update_dict[key] = (local_state[key].detach().float() - snapshot[key]).to(device)
            dict_form_updates.append(update_dict)
        
        # Step 2: Vectorize updates for norm computation
        updates_vec = []
        for update_dict in dict_form_updates:
            vec = self._vectorize_update(update_dict, model_keys)
            updates_vec.append(vec)
        
        if not updates_vec or len(updates_vec[0]) == 0:
            return global_model
        
        updates_matrix = np.array(updates_vec)  # [num_clients, num_params]
        
        # Step 3: Norm clipping based on median norm
        client_norms = np.linalg.norm(updates_matrix, axis=1)
        median_norm = np.median(client_norms)
        
        # Handle edge case: if median_norm is 0 or very small
        if median_norm < 1e-8:
            # If all updates are near zero, use them as-is
            grad_clipped = updates_matrix.copy()
        else:
            grads_clipped_norm = np.clip(client_norms, a_min=0, a_max=median_norm)
            # Clip updates (handle division by zero)
            client_norms_safe = np.where(client_norms > 1e-8, client_norms, 1.0)
            grad_clipped = (updates_matrix / client_norms_safe.reshape(-1, 1)) * grads_clipped_norm.reshape(-1, 1)
        
        # Convert back to dict form
        dict_form_grad_clipped = []
        for i in range(num_clients):
            clipped_dict = self._unvectorize_update(grad_clipped[i], model_keys, snapshot)
            # Move to device
            for key in clipped_dict.keys():
                clipped_dict[key] = clipped_dict[key].to(device)
            dict_form_grad_clipped.append(clipped_dict)
        
        # Step 4: Sparsify each client's update
        for i in range(num_clients):
            dict_form_updates[i] = self._sparse_update(dict_form_updates[i], model_keys)
        
        # Step 5: Layer-wise filtering and aggregation
        key_mean_weight = {}
        
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            
            # Get flattened updates for this layer
            key_flattened_updates = np.array([
                dict_form_updates[i][key].detach().cpu().numpy().flatten()
                for i in range(num_clients)
            ])
            
            # Step 5a: Norm filtering using MZ-score
            grad_l2norm = np.linalg.norm(key_flattened_updates, axis=1)
            S1_benign_idx = self._mz_score(grad_l2norm, self.norm_bound)
            
            # Step 5b: Sign filtering using MZ-score
            layer_signs = np.empty(num_clients)
            for i in range(num_clients):
                sign_feat = np.sign(dict_form_updates[i][key].detach().cpu().numpy())
                # Compute sign agreement metric: 0.5 * sum(sign) / sum(|sign|) * (1 - sparsity)
                layer_signs[i] = 0.5 * np.sum(sign_feat) / (np.sum(np.abs(sign_feat)) + 1e-8) * (1 - self.sparsity)
            
            S2_benign_idx = self._mz_score(layer_signs, self.sign_bound)
            
            # Step 5c: Take intersection of benign clients
            benign_idx = list(set(S1_benign_idx.tolist()).intersection(set(S2_benign_idx.tolist())))
            
            # Fallback: use all clients if no benign clients found
            if len(benign_idx) == 0:
                benign_idx = list(range(num_clients))
                print(f"🔍 LASAAggregation: Warning - no benign clients found for layer {key}, using all clients")
            
            # Step 5d: Aggregate norm-clipped updates from benign clients
            benign_updates = [dict_form_grad_clipped[i][key] for i in benign_idx]
            if len(benign_updates) > 0:
                # Ensure all updates are on the same device
                benign_updates_tensor = torch.stack([u.to(device) for u in benign_updates])
                key_mean_weight[key] = benign_updates_tensor.mean(dim=0)
            else:
                # Fallback: use zero update if no benign clients (should not happen due to fallback above)
                key_mean_weight[key] = torch.zeros_like(snapshot[key], device=device)
        
        # Step 6: Apply aggregated updates to global model
        with torch.no_grad():
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                if key in key_mean_weight:
                    # Ensure dtype compatibility
                    update = key_mean_weight[key].to(global_state[key].dtype)
                    global_state[key] = snapshot[key] + update
            
            # Keep batch norm tracking parameters from first client
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    if key in client_results[0]['model_state']:
                        global_state[key] = client_results[0]['model_state'][key]
        
        global_model.load_state_dict(global_state)
        return global_model

class BucketingAggregation(BaseAggregation):
    """Bucketing: Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing.
    
    Bucketing aggregates updates by:
    1. Randomly shuffling client updates
    2. Dividing updates into buckets of fixed size
    3. Averaging updates within each bucket
    4. Applying a selected aggregator (e.g., Krum, Median) on bucket averages
    
    This method is particularly effective for heterogeneous datasets and provides
    an additional layer of robustness by first reducing the number of updates
    through bucketing before applying the main aggregation method.
    
    Steps per round:
      1) Shuffle client results randomly
      2) Divide clients into buckets of size bucket_size
      3) Average updates within each bucket (create bucket representatives)
      4) Apply selected_aggregator on bucket averages
    
    Config keys (under params):
      - bucket_size: int (default 2) - size of each bucket (normally 2, 5, or 10)
      - selected_aggregator: str (default "Krum") - aggregator to use on bucket averages
        (can be "Krum", "MultiKrum", "Median", "FedAvg", etc.)
        # agg_config = { 'name': 'Bucketing', 'params': { 'bucket_size': 2, 'selected_aggregator': 'Krum', 'selected_aggregator_params': { # Parameters for Krum if needed 'k': 1, 'f': 1 } } }
        # agg_config = { 'name': 'Bucketing', 'params': { 'bucket_size': 2, 'selected_aggregator': 'Median', 'selected_aggregator_params': {} } }
        # agg_config = { 'name': 'Bucketing', 'params': { 'bucket_size': 5, 'selected_aggregator': 'MultiKrum', 'selected_aggregator_params': { 'avg_percentage': 0.2, 'enable_check': False } } }
      - selected_aggregator_params: dict (default {}) - parameters for the selected aggregator
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.bucket_size = int(self.params.get('bucket_size', 5))
        self.selected_aggregator = str(self.params.get('selected_aggregator', 'Median'))
        self.selected_aggregator_params = self.params.get('selected_aggregator_params', {})
        self._nested_aggregator = None  # Lazy initialization to avoid circular dependency
        
        print(f"🔍 BucketingAggregation: bucket_size: {self.bucket_size}, selected_aggregator: {self.selected_aggregator}")
    
    def _get_nested_aggregator(self):
        """Lazy initialization of nested aggregator to avoid circular dependency."""
        if self._nested_aggregator is None:
            nested_agg_config = {
                'name': self.selected_aggregator,
                'params': self.selected_aggregator_params
            }
            # Use global function (defined later in the file, but available at runtime)
            # This avoids circular dependency since the function is called lazily
            self._nested_aggregator = create_aggregation_method(nested_agg_config)
        return self._nested_aggregator
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using Bucketing method."""
        if not client_results:
            return global_model
        
        # Step 1: Shuffle client results randomly (matching original Bucketing)
        shuffled_results = client_results.copy()
        random.shuffle(shuffled_results)
        
        # Step 2: Divide into buckets
        num_buckets = math.ceil(len(shuffled_results) / self.bucket_size)
        buckets = []
        for i in range(0, len(shuffled_results), self.bucket_size):
            bucket = shuffled_results[i:i + self.bucket_size]
            buckets.append(bucket)
        
        print(f"🔍 BucketingAggregation: {len(shuffled_results)} clients divided into {num_buckets} buckets")
        
        # Step 3: Average updates within each bucket
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        device = next(global_model.parameters()).device
        
        bucket_results = []
        for bucket_id, bucket in enumerate(buckets):
            if not bucket:
                continue
            
            # Compute average model state for this bucket
            # Original uses np.mean (simple average), not weighted average
            bucket_model_state = {}
            total_samples = sum(result['samples'] for result in bucket)
            
            # Simple average of model states in the bucket (matching original: np.mean)
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    # Use the value from the first client in the bucket
                    bucket_model_state[key] = bucket[0]['model_state'][key]
                    continue
                
                # Simple average (matching original np.mean behavior)
                bucket_sum = torch.zeros_like(snapshot[key], dtype=torch.float32)
                for result in bucket:
                    bucket_sum += result['model_state'][key].float()
                
                bucket_model_state[key] = (bucket_sum / len(bucket)).to(device)
            
            # Create a bucket result (representing the averaged bucket)
            bucket_result = {
                'model_state': bucket_model_state,
                'samples': total_samples  # Sum of samples in the bucket
            }
            bucket_results.append(bucket_result)
        
        if not bucket_results:
            return global_model
        
        # Step 4: Apply selected aggregator on bucket averages
        # Note: In the original implementation, it uses np.mean on flattened updates
        # and then applies the aggregator. We're working with state dicts, so we
        # create bucket results and apply the aggregator directly.
        nested_aggregator = self._get_nested_aggregator()
        aggregated_model = nested_aggregator.aggregate(
            global_model, 
            bucket_results, 
            round_num
        )
        
        return aggregated_model

class AURORAggregation(BaseAggregation):
    """AUROR: Defending against poisoning attacks in collaborative deep learning systems.
    
    AUROR clusters coordinate values of feature vectors into 2 clusters and determines
    indicative features by checking the distance between cluster centers. Then, it
    clusters the indicative features to get the majority cluster as benign ones for
    aggregation.
    
    Steps per round:
      1) For first indicative_find_epoch epochs:
         a) Extract output layer updates (last 2 layers: weight and bias)
         b) For each feature coordinate, cluster values across clients into 2 clusters
         c) If distance between cluster centers >= threshold, mark as indicative feature
         d) Store indicative feature indices
      2) Use indicative features to cluster clients into 2 groups
      3) Select majority cluster as benign
      4) Aggregate only benign clients
    
    Config keys (under params):
      - indicative_threshold: float (default 0.002) - threshold for selecting indicative features
        (suggested: 1e-4 for MNIST LeNet5, 7e-4 for CIFAR10 ResNet18)
      - indicative_find_epoch: int (default 10) - number of epochs to find indicative features
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.indicative_threshold = float(self.params.get('indicative_threshold', 0.002))
        self.indicative_find_epoch = int(self.params.get('indicative_find_epoch', 10))
        self.epoch_cnt = 0
        self.indicative_idx = []  # Store indicative feature indices
        
        print(f"🔍 AURORAggregation: indicative_threshold: {self.indicative_threshold}, indicative_find_epoch: {self.indicative_find_epoch}")
    
    def _vectorize_model(self, model_state: Dict[str, torch.Tensor], 
                        model_keys: List[str]) -> np.ndarray:
        """Vectorize model state into a flat numpy array."""
        vec_list = []
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(model_state[key].detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])
    
    def _extract_output_layer_updates(self, updates_vec: List[np.ndarray], 
                                      model_keys: List[str],
                                      global_snapshot: Dict[str, torch.Tensor]) -> np.ndarray:
        """Extract output layer updates (last 2 layers: weight and bias).
        
        Matching original ol_from_vector logic:
        - Last 2 layers in state_dict are output layer weight and bias
        - Extract these from the end of the vectorized model
        
        Args:
            updates_vec: List of vectorized updates [num_clients, num_params]
            model_keys: List of model keys
            global_snapshot: Global model snapshot for shape reference
            
        Returns:
            Output layer updates [num_clients, output_layer_dim]
        """
        # Find output layer keys (last 2 layers)
        output_layer_keys = []
        for key in reversed(model_keys):
            if 'num_batches_tracked' not in key:
                output_layer_keys.append(key)
                if len(output_layer_keys) >= 2:
                    break
        
        if len(output_layer_keys) < 2:
            # Fallback: use last layer if only one found
            output_layer_keys = [key for key in reversed(model_keys) if 'num_batches_tracked' not in key][:2]
        
        # Calculate sizes of output layer parameters
        weight_key = output_layer_keys[0]
        bias_key = output_layer_keys[1] if len(output_layer_keys) > 1 else None
        
        weight_size = global_snapshot[weight_key].numel()
        bias_size = global_snapshot[bias_key].numel() if bias_key else 0
        
        # Extract output layer from end of vector (matching ol_from_vector)
        ol_updates = []
        for update_vec in updates_vec:
            if bias_size > 0:
                # Extract bias (last bias_size elements)
                bias = update_vec[-bias_size:]
                # Extract weight (before bias)
                weight = update_vec[-(bias_size + weight_size):-bias_size]
                # Concatenate weight and bias (matching ol_from_vector return_type='vector')
                ol_update = np.concatenate([weight.flatten(), bias.flatten()])
            else:
                # Only weight
                weight = update_vec[-weight_size:]
                ol_update = weight.flatten()
            ol_updates.append(ol_update)
        
        return np.array(ol_updates)
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using AUROR method."""
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        device = next(global_model.parameters()).device
        num_clients = len(client_results)
        
        # Step 1: Compute client updates (deltas)
        updates_vec = []
        for result in client_results:
            update_dict = {}
            local_state = result['model_state']
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                update_dict[key] = (local_state[key].detach().float() - snapshot[key]).to(device)
            
            # Vectorize update
            update_vec = self._vectorize_model(update_dict, model_keys)
            updates_vec.append(update_vec)
        
        if not updates_vec or len(updates_vec[0]) == 0:
            return global_model
        
        updates_matrix = np.array(updates_vec)  # [num_clients, num_params]
        
        # Step 2: Find indicative features (for first indicative_find_epoch epochs)
        if self.epoch_cnt < self.indicative_find_epoch:
            self.indicative_idx = []
            
            # Extract output layer updates
            ol_updates = self._extract_output_layer_updates(updates_vec, model_keys, snapshot)
            
            if ol_updates.size > 0:
                feature_dim = ol_updates.shape[1]
                
                # For each feature coordinate, cluster values across clients
                for feature_idx in range(feature_dim):
                    feature_arr = ol_updates[:, feature_idx]
                    
                    # Cluster into 2 groups using KMeans
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
                    kmeans.fit(feature_arr.reshape(-1, 1))
                    centers = kmeans.cluster_centers_
                    
                    # Check distance between cluster centers
                    center_distance = abs(centers[0][0] - centers[1][0])
                    
                    if center_distance >= self.indicative_threshold:
                        self.indicative_idx.append(feature_idx)
                
                # Convert output layer indices to full model indices
                # Indicative indices are in output layer space, need to map to full model space
                if len(self.indicative_idx) > 0:
                    # Output layer is at the end of the vector
                    ol_start_idx = len(updates_vec[0]) - ol_updates.shape[1]
                    self.indicative_idx = np.array(self.indicative_idx, dtype=np.int64) + ol_start_idx
                else:
                    self.indicative_idx = np.array([], dtype=np.int64)
            else:
                self.indicative_idx = np.array([], dtype=np.int64)
        
        # Step 3: Cluster indicative features for anomaly detection
        if len(self.indicative_idx) == 0:
            # If no indicative features found, use all features
            print("🔍 AURORAggregation: Warning - no indicative features found, using all features")
            indicative_updates = updates_matrix
        else:
            # Extract indicative features
            indicative_updates = updates_matrix[:, self.indicative_idx]
        
        # Cluster clients into 2 groups using indicative features
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(indicative_updates)
        labels = kmeans.labels_
        
        # Select majority cluster as benign
        # Count clients in each cluster
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        # Find label with most clients
        max_count_idx = np.argmax(label_counts)
        benign_label = unique_labels[max_count_idx]
        
        print(f"🔍 AURORAggregation: epoch {self.epoch_cnt}, indicative_features: {len(self.indicative_idx)}, benign_label: {benign_label}, label_counts: {label_counts}")
        
        # Step 4: Aggregate only benign clients
        benign_indices = np.where(labels == benign_label)[0]
        
        if len(benign_indices) == 0:
            # Fallback: use all clients if no benign clients found
            print("🔍 AURORAggregation: Warning - no benign clients found, using all clients")
            benign_indices = np.arange(num_clients)
        
        # Aggregate benign clients using simple average
        benign_updates = updates_matrix[benign_indices]
        aggregated_update_vec = np.mean(benign_updates, axis=0)
        
        # Convert back to state dict and apply to global model
        aggregated_update_dict = {}
        offset = 0
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            numel = snapshot[key].numel()
            update_tensor = torch.from_numpy(
                aggregated_update_vec[offset:offset + numel].reshape(snapshot[key].shape)
            ).float().to(device)
            aggregated_update_dict[key] = update_tensor
            offset += numel
        
        # Apply aggregated update to global model
        with torch.no_grad():
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                if key in aggregated_update_dict:
                    update = aggregated_update_dict[key].to(global_state[key].dtype)
                    global_state[key] = snapshot[key] + update
            
            # Keep batch norm tracking parameters from first benign client
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    if key in client_results[0]['model_state']:
                        global_state[key] = client_results[0]['model_state'][key]
        
        self.epoch_cnt += 1
        global_model.load_state_dict(global_state)
        return global_model

class SignGuardAggregation(BaseAggregation):
    """SignGuard: Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering.
    
    SignGuard filters benign clients using:
    1. Norm-based filtering: Filters clients based on lower and upper bounds relative to median norm
    2. Sign-based clustering: Clusters clients based on sign statistics of randomly selected coordinates
    3. Norm clipping: Clips benign gradients by median norm
    4. Aggregation: Aggregates clipped benign gradients
    
    Steps per round:
      1) Filter clients based on norm (lower_bound * median < norm < upper_bound * median)
      2) Randomly select a fraction of coordinates
      3) Extract sign features (positive, zero, negative ratios) from selected coordinates
      4) Cluster clients based on sign features using selected clustering algorithm
      5) Select majority cluster as benign
      6) Take intersection of norm-filtered and sign-clustered clients
      7) Clip benign gradients by median norm
      8) Aggregate clipped gradients
    
    Config keys (under params):
      - lower_bound: float (default 0.1) - lower bound multiplier for norm filtering
      - upper_bound: float (default 3.0) - upper bound multiplier for norm filtering
      - selection_fraction: float (default 0.1) - fraction of coordinates to select for clustering
      - clustering: str (default "DBSCAN") - clustering algorithm ("DBSCAN", "KMeans", or "MeanShift")
      - random_seed: int (default 0) - random seed for reproducibility
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.params = config.get('params', {})
        self.lower_bound = float(self.params.get('lower_bound', 0.1))
        self.upper_bound = float(self.params.get('upper_bound', 3.0))
        self.selection_fraction = float(self.params.get('selection_fraction', 0.1))
        self.clustering = str(self.params.get('clustering', 'DBSCAN'))
        self.random_seed = int(self.params.get('random_seed', 0))
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        print(f"🔍 SignGuardAggregation: lower_bound: {self.lower_bound}, upper_bound: {self.upper_bound}, "
              f"selection_fraction: {self.selection_fraction}, clustering: {self.clustering}")
    
    def _vectorize_updates(self, updates_dict: List[Dict[str, torch.Tensor]], 
                          model_keys: List[str]) -> np.ndarray:
        """Vectorize update dictionaries into a matrix."""
        updates_vec = []
        for update_dict in updates_dict:
            vec_list = []
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                vec_list.append(update_dict[key].detach().cpu().numpy().flatten())
            updates_vec.append(np.concatenate(vec_list) if vec_list else np.array([]))
        return np.array(updates_vec)
    
    def _norm_filtering(self, gradient_updates: np.ndarray) -> tuple:
        """Filter clients based on norm of updates.
        
        Args:
            gradient_updates: Gradient updates matrix [num_clients, num_params]
            
        Returns:
            Tuple of (benign_indices, median_norm, client_norms)
        """
        client_norms = np.linalg.norm(gradient_updates, axis=1)
        median_norm = np.median(client_norms)
        
        # Filter clients with norm within bounds
        lower_threshold = self.lower_bound * median_norm
        upper_threshold = self.upper_bound * median_norm
        benign_idx = np.argwhere(
            (client_norms > lower_threshold) & (client_norms < upper_threshold)
        )
        
        return benign_idx.reshape(-1).tolist(), median_norm, client_norms
    
    def _sign_clustering(self, gradient_updates: np.ndarray) -> List[int]:
        """Cluster clients based on sign statistics of randomly selected coordinates.
        
        Args:
            gradient_updates: Gradient updates matrix [num_clients, num_params]
            
        Returns:
            List of benign client indices
        """
        num_clients, num_para = gradient_updates.shape
        
        # Step 1: Randomized coordinate selection
        num_selected = int(self.selection_fraction * num_para)
        max_start_idx = max(0, int((1 - self.selection_fraction) * num_para) - 1)
        start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        
        # Extract selected coordinates
        randomized_weights = gradient_updates[:, start_idx:(start_idx + num_selected)]
        
        # Step 2: Extract sign statistics
        sign_grads = np.sign(randomized_weights)
        sign_type = {"pos": 1, "zero": 0, "neg": -1}
        
        def sign_feat(sign_type_val):
            """Compute normalized sign feature."""
            sign_f = (sign_grads == sign_type_val).sum(axis=1, dtype=np.float32) / num_selected
            # Normalize by max (matching original)
            max_val = sign_f.max() if sign_f.max() > 0 else 1.0
            return sign_f / (max_val + 1e-8)
        
        # Create sign features matrix [num_clients, 3]
        sign_features = np.empty((num_clients, 3), dtype=np.float32)
        sign_features[:, 0] = sign_feat(sign_type["pos"])
        sign_features[:, 1] = sign_feat(sign_type["zero"])
        sign_features[:, 2] = sign_feat(sign_type["neg"])
        
        # Step 3: Clustering based on sign statistics
        if self.clustering == "MeanShift":
            bandwidth = estimate_bandwidth(sign_features, quantile=0.5, n_samples=min(50, num_clients))
            sign_cluster = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
        elif self.clustering == "DBSCAN":
            sign_cluster = DBSCAN(eps=0.05, min_samples=3)
        elif self.clustering == "KMeans":
            sign_cluster = KMeans(n_clusters=2, random_state=self.random_seed, n_init=10)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.clustering}")
        
        sign_cluster.fit(sign_features)
        labels = sign_cluster.labels_
        
        # Step 4: Select the cluster with the majority of clients
        unique_labels = np.unique(labels)
        # Remove noise label (-1) if present
        cluster_labels = unique_labels[unique_labels != -1]
        n_cluster = len(cluster_labels)
        
        if n_cluster == 0:
            # Fallback: use all clients if no clusters found
            print("🔍 SignGuardAggregation: Warning - no clusters found, using all clients")
            return list(range(num_clients))
        
        # Find cluster with most clients
        cluster_counts = [np.sum(labels == label) for label in cluster_labels]
        max_count_idx = np.argmax(cluster_counts)
        benign_label = cluster_labels[max_count_idx]
        
        benign_idx = [int(idx) for idx in np.argwhere(labels == benign_label).flatten()]
        return benign_idx
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using SignGuard method."""
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        device = next(global_model.parameters()).device
        num_clients = len(client_results)
        
        # Step 1: Compute client updates (deltas)
        updates_dict = []
        for result in client_results:
            update_dict = {}
            local_state = result['model_state']
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                update_dict[key] = (local_state[key].detach().float() - snapshot[key]).to(device)
            updates_dict.append(update_dict)
        
        # Step 2: Vectorize updates
        gradient_updates = self._vectorize_updates(updates_dict, model_keys)
        
        if gradient_updates.size == 0 or gradient_updates.shape[0] == 0:
            return global_model
        
        # Step 3: Norm-based filtering
        S1_benign_idx, median_norm, client_norms = self._norm_filtering(gradient_updates)
        
        # Step 4: Sign-based clustering
        S2_benign_idx = self._sign_clustering(gradient_updates)
        
        # Step 5: Take intersection of norm-filtered and sign-clustered clients
        benign_idx = list(set(S1_benign_idx).intersection(set(S2_benign_idx)))
        
        if len(benign_idx) == 0:
            # Fallback: use norm-filtered clients if intersection is empty
            print("🔍 SignGuardAggregation: Warning - no clients in intersection, using norm-filtered clients")
            benign_idx = S1_benign_idx if len(S1_benign_idx) > 0 else list(range(num_clients))
        
        print(f"🔍 SignGuardAggregation: norm_filtered: {len(S1_benign_idx)}, "
              f"sign_clustered: {len(S2_benign_idx)}, benign: {len(benign_idx)}")
        
        # Step 6: Clip benign gradients by median norm
        grads_clipped_norm = np.clip(
            client_norms[benign_idx], a_min=0, a_max=median_norm
        )
        
        # Clip updates
        benign_gradient_updates = gradient_updates[benign_idx]
        client_norms_benign = client_norms[benign_idx]
        
        # Handle division by zero
        client_norms_safe = np.where(client_norms_benign > 1e-8, client_norms_benign, 1.0)
        benign_clipped = (
            benign_gradient_updates / client_norms_safe.reshape(-1, 1)
        ) * grads_clipped_norm.reshape(-1, 1)
        
        # Step 7: Aggregate clipped gradients (simple average)
        aggregated_update_vec = np.mean(benign_clipped, axis=0)
        
        # Step 8: Convert back to state dict and apply to global model
        aggregated_update_dict = {}
        offset = 0
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            numel = snapshot[key].numel()
            update_tensor = torch.from_numpy(
                aggregated_update_vec[offset:offset + numel].reshape(snapshot[key].shape)
            ).float().to(device)
            aggregated_update_dict[key] = update_tensor
            offset += numel
        
        # Apply aggregated update to global model
        with torch.no_grad():
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                if key in aggregated_update_dict:
                    update = aggregated_update_dict[key].to(global_state[key].dtype)
                    global_state[key] = snapshot[key] + update
            
            # Keep batch norm tracking parameters from first benign client
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    if key in client_results[0]['model_state']:
                        global_state[key] = client_results[0]['model_state'][key]
        
        global_model.load_state_dict(global_state)
        return global_model

class MeanAggregation(BaseAggregation):
    """Mean: Simple average aggregation (equal weights for all clients).
    
    Mean aggregation computes the simple average of all client updates,
    giving equal weight to each client regardless of their sample size.
    This is different from FedAvg which uses weighted averaging based on
    the number of samples each client has.
    
    Steps per round:
      1) Compute client updates (deltas from global model)
      2) Average all updates with equal weights
      3) Apply aggregated update to global model
    
    This method is equivalent to FedAvg when all clients have the same
    number of samples, but uses equal weights instead of sample-weighted
    averaging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        print("🔍 MeanAggregation: Using simple average (equal weights)")
    
    def aggregate(self, global_model: nn.Module, client_results: List[Dict], 
                  round_num: int) -> nn.Module:
        """Aggregate client updates using simple mean (equal weights)."""
        if not client_results:
            return global_model
        
        model_keys = list(client_results[0]['model_state'].keys())
        global_state = global_model.state_dict()
        snapshot = {k: v.clone().detach().float() for k, v in global_state.items()}
        device = next(global_model.parameters()).device
        
        # Compute client updates (deltas)
        updates_list = []
        for result in client_results:
            update_dict = {}
            local_state = result['model_state']
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                update_dict[key] = (local_state[key].detach().float() - snapshot[key]).to(device)
            updates_list.append(update_dict)
        
        # Simple average (equal weights for all clients)
        num_clients = len(updates_list)
        aggregated_update = {}
        
        for key in model_keys:
            if 'num_batches_tracked' in key:
                continue
            
            # Sum all updates
            update_sum = torch.zeros_like(snapshot[key], dtype=torch.float32)
            for update_dict in updates_list:
                update_sum += update_dict[key].float()
            
            # Average (equal weights)
            aggregated_update[key] = (update_sum / num_clients).to(device)
        
        # Apply aggregated update to global model
        with torch.no_grad():
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    continue
                if key in aggregated_update:
                    update = aggregated_update[key].to(global_state[key].dtype)
                    global_state[key] = snapshot[key] + update
            
            # Keep batch norm tracking parameters from first client
            for key in model_keys:
                if 'num_batches_tracked' in key:
                    if key in client_results[0]['model_state']:
                        global_state[key] = client_results[0]['model_state'][key]
        
        global_model.load_state_dict(global_state)
        return global_model

def create_aggregation_method(agg_config: Dict[str, Any]) -> BaseAggregation:
    """Factory function to create aggregation method instances"""
    method_name = agg_config['name']
    
    if method_name == 'FedAvg':
        return FedAvgAggregation(agg_config)
    elif method_name == 'FedSGD':
        return FedSGDAggregation(agg_config)
    elif method_name == 'FedProx' or method_name == 'fedprox':
        return FedProxAggregation(agg_config)
    elif method_name == 'SCAFFOLD' or method_name == 'scaffold':
        return SCAFFOLDAggregation(agg_config)
    elif method_name == 'FedOpt' or method_name == 'fedopt':
        return FedOptAggregation(agg_config)
    elif method_name == 'Mean' or method_name == 'mean':
        return MeanAggregation(agg_config)
    elif method_name == 'Median':
        return MedianAggregation(agg_config)
    elif method_name == 'CRFL':
        return CRFLAggregation(agg_config)
    elif method_name in ['NormClipping', 'WeakDP']: # WeakDP = NormClipping + Differential Privacy
        return NormClippingAggregation(agg_config)
    elif method_name == 'CenteredClipping':
        return CenteredClippingAggregation(agg_config)
    elif method_name == 'Flame':
        return FlameAggregation(agg_config)
    elif method_name == 'DeepSight':
        return DeepSightAggregation(agg_config)
    elif method_name == 'RFA':
        return RFAAggregation(agg_config)
    

    elif method_name in ['Krum', 'MultiKrum']:
        return KrumLikeAggregation(agg_config) # Krum or MultiKrum
    elif method_name in ['CoordinateWiseMedian']:
        return CoordinateWiseMedianAggregation(agg_config)
    elif method_name == 'TrimmedMean':
        return TrimmedMeanAggregation(agg_config)
    elif method_name in ['Bulyan']:
        return BulyanAggregation(agg_config)
    
    elif method_name == 'SimpleClustering':
        return SimpleClusteringAggregation(agg_config)
    elif method_name == 'FoolsGold':
        return FoolsGoldAggregation(agg_config)
    elif method_name == 'FLTrust':
        return FLTrustAggregation(agg_config)
    elif method_name == 'FLDetector':
        return FLDetectorAggregation(agg_config)
    elif method_name == 'RLR':
        return RLRAggregation(agg_config)
    elif method_name == 'MultiMetric' or method_name == 'mm':
        return MultiMetricAggregation(agg_config)
    elif method_name == 'DnC' or method_name == 'dnc':
        return DnCAggregation(agg_config)
    elif method_name == 'FLARE' or method_name == 'flare':
        return FLAREAggregation(agg_config)
    elif method_name == 'LASA' or method_name == 'lasa':
        return LASAAggregation(agg_config)
    elif method_name == 'Bucketing' or method_name == 'bucketing':
        return BucketingAggregation(agg_config)
    elif method_name == 'AUROR' or method_name == 'Auror' or method_name == 'auror':
        return AURORAggregation(agg_config)
    elif method_name == 'SignGuard' or method_name == 'signguard':
        return SignGuardAggregation(agg_config)
    else:
        raise ValueError(f"Unknown aggregation method: {method_name}")
