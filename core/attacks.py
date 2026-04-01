"""
Attack implementations for federated learning
"""

from numpy.random import poisson
import torch
import torchvision
import os
import numpy as np
from typing import Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pickle
import copy
import torch.nn as nn
import math

def dump_poisoned_images(clean_imgs: torch.Tensor, clean_labels: torch.Tensor,
                        poison_indices: torch.Tensor,
                        poisoned_imgs: torch.Tensor, poisoned_labels: torch.Tensor,
                        noise_imgs: torch.Tensor,
                        attack_config: Dict[str, Any]) -> None:
    """Dump poisoned images visualization to PNG file with labels"""
    def norm_img(x):
        return torch.clamp(x, 0.0, 1.0)

    clean_imgs = norm_img(clean_imgs)
    poisoned_imgs = norm_img(poisoned_imgs)
    noise_imgs = norm_img(noise_imgs)
    noise_imgs = norm_img((noise_imgs - noise_imgs.min()) / (noise_imgs.max() - noise_imgs.min() + 1e-8))

    dump_path = attack_config['dump_path']
    os.makedirs(dump_path, exist_ok=True)

    n_show = min(9, clean_imgs.size(0))

    clean_imgs = clean_imgs.cpu()
    poisoned_imgs = poisoned_imgs.cpu()
    noise_imgs = noise_imgs.cpu()

    dataset_name = attack_config['dataset_name']
    attack_name = attack_config['name']

    eps = attack_config.get('atk_eps', -1)

    if clean_labels is not None and poisoned_labels is not None:
        clean_np = clean_imgs.permute(0, 2, 3, 1).cpu().numpy()
        poisoned_np = poisoned_imgs.permute(0, 2, 3, 1).cpu().numpy()
        noise_np = noise_imgs.permute(0, 2, 3, 1).cpu().numpy()

        fig, axes = plt.subplots(3, n_show, figsize=(n_show * 2, 6))
        if n_show == 1:
            axes = axes.reshape(3, 1)

        row_labels = ['Clean Images', 'Poisoned Images', 'Residue']

        for i in range(3):
            for j in range(n_show):
                ax = axes[i, j]
                if i == 0:
                    img_data = clean_np[j]
                    label_text = f"{clean_labels[j].item()}"
                    label_color = 'blue'
                elif i == 1:
                    img_data = poisoned_np[j]
                    label_text = f"{poisoned_labels[j].item()}"
                    label_color = 'red'
                elif i == 2:
                    img_data = noise_np[j]
                    label_text = ""
                    label_color = 'black'

                if img_data.shape[-1] == 1:
                    ax.imshow(img_data.squeeze(-1), cmap='gray')
                else:
                    ax.imshow(img_data)
                ax.axis('off')

                if label_text:
                    label_len = len(label_text)
                    if label_len == 1:
                        xpos = 0.88
                    elif label_len == 2:
                        xpos = 0.80
                    else:
                        xpos = 0.72
                else:
                    xpos = 0.88
                if label_text:
                    ax.text(xpos, 0.96, label_text, transform=ax.transAxes,
                           fontsize=14, fontweight='bold', color=label_color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                                   edgecolor=label_color, alpha=0.8),
                           verticalalignment='top', horizontalalignment='left')

            axes[i, 0].text(-0.15, 0.5, row_labels[i], transform=axes[i, 0].transAxes,
                           fontsize=12, fontweight='bold', rotation=90,
                           verticalalignment='center', horizontalalignment='center')

        plt.tight_layout()
        save_path = os.path.join(dump_path, f'{dataset_name}_{attack_name}_with_labels_{attack_config.get("target_class", -1)}_eps_{eps}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ Saved poisoned images with labels (3-row grid) to {save_path}")
        plt.close()

    target_class = attack_config['target_class']
    eps = attack_config.get('atk_eps', -1)
    latent_dim = attack_config.get('atk_latent_dim', -1)
    fn = f'{dataset_name}_{attack_name}_labels_{target_class}_eps_{eps}_latent_dim_{latent_dim}_images.pickle'
    with open(os.path.join(dump_path, fn), 'wb') as f:
        pickle.dump({'clean_imgs': clean_np, 'poisoned_imgs': poisoned_np, 'noise_imgs': noise_np,
                    'target_class': target_class, 'eps': eps, 'latent_dim': latent_dim,
                    'clean_labels': clean_labels.cpu().numpy(), 'poisoned_labels': poisoned_labels.cpu().numpy()}, f)
    print(f"🖼️ Saved all images to to .pickle file {os.path.join(dump_path, fn)}")


class BaseAttack(ABC):
    """Base class for all attacks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.poison_ratio = config.get('poison_ratio', 0.5)
        self.attack_start_round = config['attack_start_round']
        self.attack_stop_round = config['attack_stop_round']
        self.attack_frequency = config['attack_frequency']
        self.mean = config['mean']
        self.std = config['std']
        self.seed = config.get('seed', 42)

    def poison_data(self, clean_data: torch.Tensor, clean_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = clean_data.shape[0]
        client_id = self.config.get('client_id', -1)
        poison_size = batch_size if client_id == -1 else int(batch_size * self.poison_ratio)

        if poison_size == 0:
            return clean_data, clean_labels

        poison_indices = torch.from_numpy(np.random.permutation(batch_size)[:poison_size])
        poisoned_data = clean_data.clone()
        poisoned_labels = clean_labels.clone()

        poisoned_data = self._denormalize(poisoned_data)
        self._apply_poison(clean_data, clean_labels, poison_indices, poisoned_data, poisoned_labels)
        poisoned_data = self._normalize(poisoned_data)

        return poisoned_data, poisoned_labels

    def _denormalize(self, data: torch.Tensor) -> torch.Tensor:
        num_channels = data.shape[1]
        if num_channels == 1:
            mean = torch.tensor(self.mean).view(1, 1, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 1, 1, 1).to(data.device)
        else:
            mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 3, 1, 1).to(data.device)
        return data * std + mean

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        num_channels = data.shape[1]
        if num_channels == 1:
            mean = torch.tensor(self.mean).view(1, 1, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 1, 1, 1).to(data.device)
        else:
            mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 3, 1, 1).to(data.device)
        return (data - mean) / std

    def _apply_poison(self, clean_data: torch.Tensor, clean_labels: torch.Tensor, poison_indices: torch.Tensor, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor) -> None:
        if self.use_model_trigger():
            model_device = next(self.atk_model.parameters()).device
            if model_device != poisoned_data.device:
                self.atk_model = self.atk_model.to(poisoned_data.device)

            self.atk_model.eval()
            with torch.no_grad():
                perturbations, target_labels = self._generate_attack_batch(poisoned_data[poison_indices], poisoned_labels[poison_indices], poisoned_data.device)

            poisoned_data[poison_indices] = perturbations
            poisoned_labels[poison_indices] = target_labels
        else:
            self._apply_static_trigger(poisoned_data, poisoned_labels, poison_indices)

    def use_model_trigger(self) -> bool:
        return hasattr(self, 'atk_model') and self.atk_model is not None

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor) -> None:
        raise NotImplementedError("Static trigger attacks must implement _apply_static_trigger()")

    def _generate_attack_batch(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_data_type(self) -> str:
        pass

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        return local_model_state

    def should_apply(self, round_idx: int) -> bool:
        if self.attack_frequency == -1:
            return round_idx >= self.attack_start_round and round_idx < self.attack_stop_round
        elif self.attack_frequency == 0:
            return False
        else:
            return round_idx >= self.attack_start_round and round_idx < self.attack_stop_round and (round_idx - self.attack_start_round) % self.attack_frequency == 0

    def _setup_training_atk_model(self, classifier: torch.nn.Module, device: str) -> Tuple[torch.optim.Optimizer, torch.nn.Module, torch.Tensor, torch.Tensor]:
        self.atk_model.to(device)
        classifier.to(device)
        self.atk_model.train()
        classifier.eval()
        print(f"[{self.__class__.__name__}] with optimizer: {self.atk_optimizer}, lr: {self.atk_lr}")
        if self.atk_optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.atk_model.parameters(), lr=self.atk_lr)
        elif self.atk_optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.atk_model.parameters(), lr=self.atk_lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.atk_optimizer}")

        loss_fn = torch.nn.CrossEntropyLoss()

        mean = torch.tensor(self.mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=device).view(1, -1, 1, 1)

        return optimizer, loss_fn, mean, std

    def _finalize_training(self, verbose: bool, client_id: int, epoch: int, local_ba: float):
        self.atk_model.eval()
        self.atk_model = self.atk_model.cpu()
        if verbose:
            print(f"[{self.__class__.__name__}] Client {client_id} Training finished at epoch {epoch}, final backdoor acc={local_ba:.4f}")

    def _train_atk_epoch_common(self, trainloader, device: str, mean: torch.Tensor, std: torch.Tensor,
                           classifier: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module) -> Tuple[float, float]:
        local_correct, total_loss, total_sample = 0, 0, 0

        for idx, batch in enumerate(trainloader):
            if isinstance(batch, dict):
                data = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)

            clean_data = data * std + mean
            atk_data, atk_label = self._generate_attack_batch(clean_data, labels, device)
            atk_data = (atk_data - mean) / std

            atk_output = classifier(atk_data)
            loss = loss_fn(atk_output, atk_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = atk_output.argmax(dim=1)
            correct = (pred == atk_label).sum().item()
            local_correct += correct
            total_sample += len(atk_label)
            total_loss += loss.item() * len(atk_label)

        avg_loss = total_loss / total_sample if total_sample > 0 else 0.0
        local_ba = local_correct / total_sample if total_sample > 0 else 0.0

        return local_ba, avg_loss


class BadNetsAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5

        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)

        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)

        self.trigger_pattern = config.get('trigger_pattern', 'square')
        self.target_class = config.get('target_class', 0)

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                             poison_indices: torch.Tensor) -> None:
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        poisoned_data[poison_indices, :, h - self.trigger_height:, w - self.trigger_width:] = 1.0
        poisoned_labels[poison_indices] = self.target_class


class BlendedAttack(BaseAttack):
    _trigger_cache = None

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.blend_alpha = config.get('blend_alpha', 0.3)
        self.target_class = config.get('target_class', 0)

    def get_data_type(self) -> str:
        return "image"

    def _get_or_create_trigger(self, device: torch.device, shape: tuple) -> torch.Tensor:
        if BlendedAttack._trigger_cache is not None and BlendedAttack._trigger_cache.shape == shape:
            return BlendedAttack._trigger_cache.to(device)

        trigger = torch.rand(shape)
        BlendedAttack._trigger_cache = trigger
        return trigger.to(device)

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                             poison_indices: torch.Tensor) -> None:
        trigger = self._get_or_create_trigger(poisoned_data.device, poisoned_data.shape[1:])

        for idx in poison_indices:
            poisoned_data[idx] = torch.clamp((1 - self.blend_alpha) * poisoned_data[idx] + self.blend_alpha * trigger, 0.0, 1.0)
            poisoned_labels[idx] = self.target_class


class SinusoidalAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 0)
        self.sine_amplitude = float(config.get('sine_amplitude', 0.2))
        self.sine_frequency = float(config.get('sine_frequency', 4.0))
        self.sine_phase = float(config.get('sine_phase', 0.0))
        self.orientation = config.get('sine_orientation', 'horizontal')
        self.channel_mode = config.get('channel_mode', 'all')
        self.channel_index = int(config.get('channel_index', 0))

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                              poison_indices: torch.Tensor) -> None:
        _, _, H, W = poisoned_data.shape

        if self.orientation == 'horizontal':
            axis_len = W
            grid = torch.linspace(0, 1, steps=axis_len, device=poisoned_data.device)
            pattern_1d = torch.sin(2 * torch.pi * self.sine_frequency * grid + self.sine_phase)
            pattern_1d = (pattern_1d + 1.0) / 2.0
            pattern = pattern_1d.view(1, 1, 1, W).expand(1, 1, H, W)
        else:
            axis_len = H
            grid = torch.linspace(0, 1, steps=axis_len, device=poisoned_data.device)
            pattern_1d = torch.sin(2 * torch.pi * self.sine_frequency * grid + self.sine_phase)
            pattern_1d = (pattern_1d + 1.0) / 2.0
            pattern = pattern_1d.view(1, 1, H, 1).expand(1, 1, H, W)

        pattern = self.sine_amplitude * pattern

        if self.channel_mode == 'single':
            num_channels = poisoned_data.shape[1]
            ch = max(0, min(self.channel_index, num_channels - 1))
            add_tensor = torch.zeros_like(poisoned_data[poison_indices])
            add_tensor[:, ch:ch+1, :, :] = pattern
            poisoned_data[poison_indices] = torch.clamp(poisoned_data[poison_indices] + add_tensor, 0.0, 1.0)
        else:
            poisoned_data[poison_indices] = torch.clamp(poisoned_data[poison_indices] + pattern, 0.0, 1.0)

        poisoned_labels[poison_indices] = self.target_class


class LabelFlippingAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.attack_model = config.get('attack_model', 'targeted')
        self.source_label = config.get('source_label', 2)
        self.target_label = config.get('target_label', 7)
        self.num_classes = config.get('num_classes', 10)

        if self.attack_model == 'targeted':
            assert self.source_label != self.target_label
            assert 0 <= self.source_label < self.num_classes
            assert 0 <= self.target_label < self.num_classes
        elif self.attack_model == 'all2one':
            assert 0 <= self.target_label < self.num_classes

    def get_data_type(self) -> str:
        return "image"

    def poison_data(self, clean_data: torch.Tensor, clean_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = clean_data.shape[0]
        client_id = self.config.get('client_id', -1)

        poisoned_data = clean_data.clone()
        poisoned_labels = clean_labels.clone()

        if self.attack_model == 'targeted':
            source_indices = (clean_labels == self.source_label).nonzero(as_tuple=True)[0]
            if len(source_indices) == 0:
                return poisoned_data, poisoned_labels

            poisoned_data = self._denormalize(poisoned_data)
            self._apply_static_trigger(poisoned_data, poisoned_labels, source_indices)
            poisoned_data = self._normalize(poisoned_data)

            return poisoned_data, poisoned_labels
        else:
            return super().poison_data(clean_data, clean_labels)

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                             poison_indices: torch.Tensor) -> None:
        if len(poison_indices) == 0:
            return

        labels_to_flip = poisoned_labels[poison_indices]

        if self.attack_model == 'targeted':
            poisoned_labels[poison_indices] = self.target_label
        elif self.attack_model == 'all2one':
            poisoned_labels[poison_indices] = self.target_label
        elif self.attack_model == 'all2all':
            inverse_labels = self.num_classes - 1 - labels_to_flip
            poisoned_labels[poison_indices] = inverse_labels
        elif self.attack_model == 'random':
            random_labels = torch.randint(0, self.num_classes, size=(len(poison_indices),),
                                         device=poisoned_labels.device, dtype=poisoned_labels.dtype)
            poisoned_labels[poison_indices] = random_labels
        else:
            raise ValueError(f"Unknown attack_model: {self.attack_model}")


class ModelReplacementAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 6)

        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5

        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)

        self.trigger_position = config.get('trigger_position', 'bottom-right')
        self.scaling_factor = config.get('scaling_factor', 50)
        self.alpha = config.get('alpha', 0.5)

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                             poison_indices: torch.Tensor) -> None:
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]

        if self.trigger_position == 'bottom-right':
            row_start = h - self.trigger_height
            row_end = h
            col_start = w - self.trigger_width
            col_end = w
        elif self.trigger_position == 'bottom-left':
            row_start = h - self.trigger_height
            row_end = h
            col_start = 0
            col_end = self.trigger_width
        else:
            raise ValueError(f"Unknown trigger_position: {self.trigger_position}")

        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        poisoned_labels[poison_indices] = self.target_class

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        scaled_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                    scaled_state[key] = local_model_state[key].clone()
                    continue

                local_param = local_model_state[key]
                global_param = global_model_state.get(key, local_param.clone())

                update = local_param.float() - global_param.float()
                scaled_param = global_param.float() + self.scaling_factor * update
                scaled_state[key] = scaled_param.to(local_param.dtype)

        return scaled_state


class NeurotoxinAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 6)

        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5

        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)

        self.trigger_position = config.get('trigger_position', 'bottom-right')
        self.topk_ratio = config.get('topk_ratio', 0.1)
        self.lambda_val = config.get('lambda_val', 5.0)

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                              poison_indices: torch.Tensor) -> None:
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        if self.trigger_position == 'bottom-right':
            row_start, row_end = h - self.trigger_height, h
            col_start, col_end = w - self.trigger_width, w
        elif self.trigger_position == 'bottom-left':
            row_start, row_end = h - self.trigger_height, h
            col_start, col_end = 0, self.trigger_width
        else:
            raise ValueError(f"Unknown trigger_position: {self.trigger_position}")

        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        poisoned_labels[poison_indices] = self.target_class

    def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        vec_list = []
        for key, param in state_dict.items():
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])

    def _unvectorize_to_state_dict(self, vector: np.ndarray,
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = {}
        offset = 0
        for key, param in reference_state.items():
            if 'num_batches_tracked' in key:
                state_dict[key] = param.clone()
                continue
            numel = param.numel()
            param_vec = vector[offset:offset + numel]
            state_dict[key] = torch.from_numpy(param_vec.reshape(param.shape)).to(param.device, dtype=param.dtype)
            offset += numel
        return state_dict

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key:
                continue
            if key in global_model_state:
                update_dict[key] = (local_model_state[key].cpu() - global_model_state[key].cpu()).float()
            else:
                update_dict[key] = local_model_state[key].cpu().clone().float()

        update_vec = self._vectorize_state_dict(update_dict)
        if len(update_vec) == 0:
            return local_model_state

        update_vec = np.nan_to_num(update_vec, nan=0.0, posinf=0.0, neginf=0.0)

        benign_norm = np.linalg.norm(update_vec)
        if benign_norm == 0:
            return local_model_state

        k = max(1, int(len(update_vec) * self.topk_ratio))
        abs_update_vec = np.abs(update_vec)
        topk_indices = np.argpartition(abs_update_vec, k)[:k]

        masked_update_vec = np.zeros_like(update_vec)
        masked_update_vec[topk_indices] = update_vec[topk_indices] * self.lambda_val

        current_norm = np.linalg.norm(masked_update_vec)

        if current_norm > benign_norm and current_norm > 0:
            scale = benign_norm / current_norm
            masked_update_vec = masked_update_vec * scale

        masked_update_vec = np.nan_to_num(masked_update_vec, nan=0.0, posinf=0.0, neginf=0.0)

        masked_update_dict = self._unvectorize_to_state_dict(masked_update_vec, local_model_state)

        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key:
                    final_state[key] = local_model_state[key].clone()
                    continue

                if key in global_model_state and key in masked_update_dict:
                    final_state[key] = global_model_state[key].cpu() + masked_update_dict[key].to(
                        global_model_state[key].dtype)
                elif key in local_model_state:
                    final_state[key] = local_model_state[key].cpu().clone()

        return final_state


class EdgeCaseBackdoorAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 1)
        self.dataset_name = config.get('dataset_name', 'mnist').lower()
        self.data_root = config.get('data_root', './data')

        self.epsilon = config.get('epsilon', 0.25)
        self.projection_type = config.get('projection_type', 'l_2')
        self.PGD_attack = config.get('PGD_attack', True)
        self.scaling_attack = config.get('scaling_attack', True)
        self.scaling_factor = config.get('scaling_factor', 50)
        self.l2_proj_frequency = config.get('l2_proj_frequency', 1)

        self.edge_case_samples = None
        self.edge_case_labels = None
        self.edge_case_idx = 0
        self._load_edge_case_samples()

    def get_data_type(self) -> str:
        return "image"

    def _load_edge_case_samples(self):
        try:
            from .edge_case_datasets import load_edge_case_dataset
            self.edge_case_samples, self.edge_case_labels = load_edge_case_dataset(
                self.dataset_name, self.target_class, self.data_root
            )
            if isinstance(self.edge_case_samples, np.ndarray):
                self.edge_case_samples = torch.from_numpy(self.edge_case_samples).float()
            if isinstance(self.edge_case_labels, np.ndarray):
                self.edge_case_labels = torch.from_numpy(self.edge_case_labels).long()
        except (FileNotFoundError, ImportError) as e:
            raise RuntimeError(f"Failed to load edge-case dataset: {e}")

    def _apply_poison(self, clean_data: torch.Tensor, clean_labels: torch.Tensor,
                      poison_indices: torch.Tensor, poisoned_data: torch.Tensor,
                      poisoned_labels: torch.Tensor) -> None:
        if self.edge_case_samples is None or len(self.edge_case_samples) == 0:
            raise RuntimeError("Edge-case samples not loaded.")

        num_to_poison = len(poison_indices)

        if self.edge_case_idx + num_to_poison <= len(self.edge_case_samples):
            edge_samples = self.edge_case_samples[self.edge_case_idx:self.edge_case_idx + num_to_poison].clone()
            edge_labels = self.edge_case_labels[self.edge_case_idx:self.edge_case_idx + num_to_poison].clone()
            self.edge_case_idx += num_to_poison
        else:
            remaining = num_to_poison
            edge_samples_list = []
            edge_labels_list = []
            while remaining > 0:
                take = min(remaining, len(self.edge_case_samples) - self.edge_case_idx)
                edge_samples_list.append(self.edge_case_samples[self.edge_case_idx:self.edge_case_idx + take].clone())
                edge_labels_list.append(self.edge_case_labels[self.edge_case_idx:self.edge_case_idx + take].clone())
                self.edge_case_idx = (self.edge_case_idx + take) % len(self.edge_case_samples)
                remaining -= take
            edge_samples = torch.cat(edge_samples_list, dim=0)
            edge_labels = torch.cat(edge_labels_list, dim=0)

        if edge_samples.dim() == 3:
            edge_samples = edge_samples.unsqueeze(1)
        elif edge_samples.dim() == 4 and edge_samples.shape[1] != 3 and edge_samples.shape[-1] == 3:
            edge_samples = edge_samples.permute(0, 3, 1, 2)

        edge_samples = torch.clamp(edge_samples, 0.0, 1.0)

        if hasattr(self, 'mean') and hasattr(self, 'std') and self.mean is not None:
            import torchvision.transforms.functional as TF
            for i in range(len(edge_samples)):
                edge_samples[i] = TF.normalize(edge_samples[i], self.mean, self.std)

        edge_samples = edge_samples.to(poisoned_data.device)
        edge_labels = edge_labels.to(poisoned_labels.device)

        poisoned_data[poison_indices] = edge_samples
        poisoned_labels[poison_indices] = edge_labels

    def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        vec_list = []
        for key, param in state_dict.items():
            if 'num_batches_tracked' in key: continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])

    def _unvectorize_to_state_dict(self, vector: np.ndarray,
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = {}
        offset = 0
        for key, param in reference_state.items():
            if 'num_batches_tracked' in key:
                state_dict[key] = param.clone()
                continue
            numel = param.numel()
            param_vec = vector[offset:offset + numel]
            state_dict[key] = torch.from_numpy(param_vec.reshape(param.shape)).to(param.device, dtype=param.dtype)
            offset += numel
        return state_dict

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key: continue
            if key in global_model_state:
                update_dict[key] = local_model_state[key] - global_model_state[key]
            else:
                update_dict[key] = local_model_state[key].clone()

        update_vec = self._vectorize_state_dict(update_dict)
        if len(update_vec) == 0: return local_model_state

        if self.PGD_attack:
            if self.projection_type == 'l_inf':
                smaller_idx = update_vec < -self.epsilon
                larger_idx = update_vec > self.epsilon
                projected_update_vec = update_vec.copy()
                projected_update_vec[smaller_idx] = -self.epsilon
                projected_update_vec[larger_idx] = self.epsilon
            elif self.projection_type == 'l_2':
                w_diff_norm = np.linalg.norm(update_vec)
                if w_diff_norm > self.epsilon:
                    projected_update_vec = self.epsilon * update_vec / w_diff_norm
                else:
                    projected_update_vec = update_vec
        else:
            projected_update_vec = update_vec

        if self.scaling_attack:
            scaled_update_vec = self.scaling_factor * projected_update_vec
        else:
            scaled_update_vec = projected_update_vec

        scaled_update_dict = self._unvectorize_to_state_dict(scaled_update_vec, local_model_state)

        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key:
                    final_state[key] = local_model_state[key].clone()
                    continue
                if key in global_model_state and key in scaled_update_dict:
                    final_state[key] = global_model_state[key] + scaled_update_dict[key]
                elif key in local_model_state:
                    final_state[key] = local_model_state[key].clone()

        return final_state


class ThreeDFedAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 0)

        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5

        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)

        self.trigger_position = config.get('trigger_position', 'bottom-right')
        self.scaling_factor = config.get('scaling_factor', 1.0)
        self.use_norm_clipping = config.get('use_norm_clipping', True)

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                             poison_indices: torch.Tensor) -> None:
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        if self.trigger_position == 'bottom-right':
            row_start = h - self.trigger_height
            row_end = h
            col_start = w - self.trigger_width
            col_end = w
        elif self.trigger_position == 'bottom-left':
            row_start = h - self.trigger_height
            row_end = h
            col_start = 0
            col_end = self.trigger_width
        else:
            raise ValueError(f"Unknown trigger_position: {self.trigger_position}")

        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        poisoned_labels[poison_indices] = self.target_class

    def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        vec_list = []
        for key, param in state_dict.items():
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])

    def _unvectorize_to_state_dict(self, vector: np.ndarray,
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_dict = {}
        offset = 0
        for key, param in reference_state.items():
            if 'num_batches_tracked' in key:
                state_dict[key] = param.clone()
                continue
            numel = param.numel()
            param_vec = vector[offset:offset + numel]
            state_dict[key] = torch.from_numpy(param_vec.reshape(param.shape)).to(param.device, dtype=param.dtype)
            offset += numel
        return state_dict

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        if not self.use_norm_clipping:
            return local_model_state

        backdoor_update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key:
                continue
            if key in global_model_state:
                backdoor_update_dict[key] = local_model_state[key] - global_model_state[key]
            else:
                backdoor_update_dict[key] = local_model_state[key].clone()

        backdoor_update_vec = self._vectorize_state_dict(backdoor_update_dict)
        if len(backdoor_update_vec) == 0:
            return local_model_state

        backdoor_norm = np.linalg.norm(backdoor_update_vec)
        reference_norm = self.config.get('reference_norm', None)

        if reference_norm is not None:
            benign_norm = reference_norm
            if backdoor_norm > benign_norm:
                backdoor_update_vec = backdoor_update_vec * (benign_norm / backdoor_norm)
                backdoor_norm = benign_norm

            scale_factor = min((benign_norm / backdoor_norm) if backdoor_norm > 0 else 1.0,
                             self.scaling_factor)
            final_scale = max(scale_factor, 1.0)
            backdoor_update_vec = backdoor_update_vec * final_scale

        clipped_update_dict = self._unvectorize_to_state_dict(backdoor_update_vec, local_model_state)

        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key:
                    final_state[key] = local_model_state[key].clone()
                    continue
                if key in global_model_state and key in clipped_update_dict:
                    final_state[key] = global_model_state[key] + clipped_update_dict[key]
                elif key in local_model_state:
                    final_state[key] = local_model_state[key].clone()

        return final_state


class DBAAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 0)
        self.trigger_nums = config.get('trigger_nums', 4)
        self.use_global_backdoor = config.get('use_global_backdoor', False)
        self.trigger_height = config.get('trigger_height', 1)
        self.trigger_width = config.get('trigger_width', 6)
        self.trigger_gap = config.get('trigger_gap', 2)
        self.trigger_shift = config.get('trigger_shift', 0)

        self.apply_to_client_ids = config.get('apply_to_client_ids', [])
        self._create_local_triggers()

    def _create_local_triggers(self):
        num_channels = len(self.mean)
        self.local_triggers = []
        for i in range(self.trigger_nums):
            if num_channels == 1:
                trigger = torch.ones((1, 1, self.trigger_height, self.trigger_width))
            else:
                trigger = torch.ones((1, num_channels, self.trigger_height, self.trigger_width))
            self.local_triggers.append(trigger)
        self._precompute_trigger_positions()

    def _precompute_trigger_positions(self):
        self.trigger_positions = []
        for i in range(self.trigger_nums):
            row_start, col_start = self._setup_trigger_position(i)
            row_end = row_start + self.trigger_height
            col_end = col_start + self.trigger_width
            self.trigger_positions.append((row_start, row_end, col_start, col_end))

    def _setup_trigger_position(self, trigger_idx: int) -> Tuple[int, int]:
        row_starter = (trigger_idx // 2) * (self.trigger_height + self.trigger_gap) + self.trigger_shift
        column_starter = (trigger_idx % 2) * (self.trigger_width + self.trigger_gap) + self.trigger_shift
        return row_starter, column_starter

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor) -> None:
        client_id = self.config.get('client_id', -1)
        if client_id == -1 or self.use_global_backdoor:
            self._apply_global_trigger(poisoned_data, poisoned_labels, poison_indices)
        else:
            self._apply_local_trigger(poisoned_data, poisoned_labels, poison_indices, client_id)

    def _apply_local_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor, client_id: int) -> None:
        trigger_idx = self.apply_to_client_ids.index(client_id) % self.trigger_nums
        trigger = self.local_triggers[trigger_idx].to(poisoned_data.device)
        row_start, row_end, col_start, col_end = self.trigger_positions[trigger_idx]

        row_end = min(row_end, poisoned_data.shape[2])
        col_end = min(col_end, poisoned_data.shape[3])

        if row_end > row_start and col_end > col_start:
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = trigger[0, :, :actual_height, :actual_width]
            poisoned_labels[poison_indices] = self.target_class

    def _apply_global_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor) -> None:
        for trigger_idx in range(self.trigger_nums):
            trigger = self.local_triggers[trigger_idx].to(poisoned_data.device)
            row_start, row_end, col_start, col_end = self.trigger_positions[trigger_idx]

            row_end = min(row_end, poisoned_data.shape[2])
            col_end = min(col_end, poisoned_data.shape[3])

            if row_end > row_start and col_end > col_start:
                actual_height = row_end - row_start
                actual_width = col_end - col_start
                poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = trigger[0, :, :actual_height, :actual_width]

        poisoned_labels[poison_indices] = self.target_class

    def get_data_type(self) -> str:
        return "image"

# # =========================================================================
# # 🚀 终极破防框架：Sign-Aligned BC-DARE (全量同向渗透 + 策略A截断)
# # 核心亮点：基于 Layer Substitution Analysis (LSA) 严格测算 BSR 下降 (含动态周期刷新)
# # =========================================================================
# class FedDAREAttack(BaseAttack):
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.target_class = config.get('target_class', 0)
#         # 严格执行全局 1% 预算 (如 0.99 则为 1%)
#         self.drop_rate = config.get('drop_rate', 0.99)
#         self.gamma = config.get('gamma', 50.0)  # 恢复 Gamma 放大，默认 50 倍
#
#         input_dim = config.get('input_dim', 32)
#         default_size = 5 if input_dim >= 32 else 4
#         self.trigger_height = config.get('trigger_height', default_size)
#         self.trigger_width = config.get('trigger_width', default_size)
#
#         # 缓存与 LSA 环境参数
#         self.lsa_model = None
#         self.lsa_dataloader = None
#         self.lsa_device = None
#         self.cached_layer_scores = None
#         self.cached_score_sum = None
#
#         # 动态 LSA 测算计数器
#         self.attack_invoke_count = 0
#         self.lsa_recompute_interval = 10  # 每  轮重测一次
#
#     def get_data_type(self) -> str:
#         return "image"
#
#     def setup_lsa_environment(self, model: torch.nn.Module, dataloader: Any, device: torch.device):
#         self.lsa_model = model
#         self.lsa_dataloader = dataloader
#         self.lsa_device = device
#
#     def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
#                               poison_indices: torch.Tensor) -> None:
#         _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
#         row_start, row_end = h - self.trigger_height, h
#         col_start, col_end = w - self.trigger_width, w
#         poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
#         poisoned_labels[poison_indices] = self.target_class
#
#     def _evaluate_bsr(self, eval_model: torch.nn.Module) -> float:
#         """【绝对严格】：跑完整个 dataloader，不准 break"""
#         correct, total = 0, 0
#         eval_model.eval()
#         with torch.no_grad():
#             for data, target in self.lsa_dataloader:
#                 data = data.to(self.lsa_device)
#                 poisoned_data = data.clone()
#                 poison_indices = torch.arange(data.shape[0])
#                 poisoned_data = self._denormalize(poisoned_data)
#                 dummy_labels = torch.zeros(data.shape[0], dtype=torch.long, device=self.lsa_device)
#                 self._apply_static_trigger(poisoned_data, dummy_labels, poison_indices)
#                 poisoned_data = self._normalize(poisoned_data)
#                 outputs = eval_model(poisoned_data)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += data.size(0)
#                 correct += (predicted == self.target_class).sum().item()
#         return correct / total if total > 0 else 0.0
#
#     def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
#                               global_model_state: Dict[str, torch.Tensor],
#                               benign_model_state: Dict[str, torch.Tensor] = None,
#                               algorithm: str = "FedAvg") -> Dict[str, torch.Tensor]:
#         if benign_model_state is None:
#             raise ValueError("Sign-Aligned BC-DARE 必须依赖 benign_model_state！")
#
#         G_poison = {}
#         G_benign = {}
#         param_shapes = {}
#         total_params = 0
#
#         for key in local_model_state.keys():
#             if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
#                 continue
#             G_poison[key] = local_model_state[key].float() - global_model_state[key].float()
#             G_benign[key] = benign_model_state[key].float() - global_model_state[key].float()
#             param_shapes[key] = G_poison[key].shape
#             total_params += G_poison[key].numel()
#
#         # =========================================================================
#         # 步骤 1：严格 LSA 打分 (动态周期刷新)
#         # =========================================================================
#         force_recompute = (self.attack_invoke_count % self.lsa_recompute_interval == 0)
#
#         if self.cached_layer_scores is None or force_recompute:
#             print(f"\n   [Sign-Aligned BC-DARE] {'首次' if not force_recompute else '周期性'}执行严格 LSA 打分...")
#             layer_scores = {}
#             score_sum = 0.0
#             self.lsa_model = self.lsa_model.to(self.lsa_device)
#             self.lsa_model.load_state_dict(local_model_state)
#             bsr_malicious = self._evaluate_bsr(self.lsa_model)
#
#             for key in G_poison.keys():
#                 hybrid_state = {k: v.clone() for k, v in local_model_state.items()}
#                 hybrid_state[key] = benign_model_state[key].clone()
#                 self.lsa_model.load_state_dict(hybrid_state)
#                 bsr_hybrid = self._evaluate_bsr(self.lsa_model)
#
#                 delta_bsr = bsr_malicious - bsr_hybrid
#                 # 严格按照“非负即投毒”：使用微小 epsilon (1e-9) 确保 0 分层也能入选
#                 score = max(1e-9, delta_bsr)
#                 layer_scores[key] = score
#                 score_sum += score
#
#                 # [加回来的打印逻辑]：只打印杀伤力大于 0 的层，避免 60 多个 0 分层刷屏
#                 if delta_bsr > 0:
#                     print(f"   [LSA] 层 {key:<30} | 真实杀伤力 (Delta BSR): {delta_bsr:.4f}")
#
#             self.cached_layer_scores = layer_scores
#             self.cached_score_sum = score_sum
#             self.lsa_model = self.lsa_model.cpu()
#         else:
#             layer_scores = self.cached_layer_scores
#             score_sum = self.cached_score_sum
#
#         # =========================================================================
#         # 步骤 2：策略 B —— 迭代式水箱平衡分配 (Iterative Water-filling)
#         # 目标：100% 耗尽预算，同时遵守“符号锁定”
#         # =========================================================================
#         global_budget = int(total_params * (1.0 - self.drop_rate))
#         remaining_budget = global_budget
#
#         # 预存所有层的“合规候选人”并排序
#         layer_candidates = {}
#         for key in G_poison.keys():
#             g_p_flat = G_poison[key].flatten().to(self.lsa_device)
#             g_b_flat = G_benign[key].flatten().to(self.lsa_device)
#
#             # [符号锁定]
#             sign_mask = (torch.sign(g_p_flat) == torch.sign(g_b_flat))
#             candidate_indices = torch.where(sign_mask)[0]
#
#             if len(candidate_indices) > 0:
#                 epsilon = 1e-8
#                 saliency = (torch.abs(g_p_flat[candidate_indices]) / (torch.abs(g_b_flat[candidate_indices]) + epsilon))
#                 sorted_inner = torch.argsort(saliency, descending=True)
#                 layer_candidates[key] = candidate_indices[sorted_inner].tolist()
#             else:
#                 layer_candidates[key] = []
#
#         # 迭代分配循环
#         final_selections = {key: [] for key in G_poison.keys()}
#         active_layers = [k for k in layer_scores.keys() if len(layer_candidates[k]) > 0]
#
#         while remaining_budget > 0 and len(active_layers) > 0:
#             current_total_score = sum(layer_scores[k] for k in active_layers)
#             layers_to_remove = []
#
#             # 记录当前剩余预算，准备分摊
#             budget_to_distribute = remaining_budget
#
#             for key in active_layers:
#                 # 严格按比例分配
#                 allocation = int(math.floor(budget_to_distribute * (layer_scores[key] / current_total_score)))
#                 # 防死锁保底
#                 if allocation == 0 and remaining_budget > 0: allocation = 1
#
#                 take = min(allocation, len(layer_candidates[key]), remaining_budget)
#
#                 if take > 0:
#                     final_selections[key].extend(layer_candidates[key][:take])
#                     layer_candidates[key] = layer_candidates[key][take:]
#                     remaining_budget -= take
#
#                 if len(layer_candidates[key]) == 0:
#                     layers_to_remove.append(key)
#
#                 if remaining_budget <= 0: break
#
#             for k in layers_to_remove: active_layers.remove(k)
#
#         # =========================================================================
#         # 步骤 3：拼装写回
#         # =========================================================================
#         poisoned_state = {}
#         for key in local_model_state.keys():
#             if key not in G_poison:
#                 poisoned_state[key] = benign_model_state[key].clone()
#                 continue
#
#             g_p_flat = G_poison[key].flatten().to(self.lsa_device)
#             g_b_flat = G_benign[key].flatten().to(self.lsa_device)
#
#             # 构造掩码
#             selection_mask = torch.zeros_like(g_p_flat)
#             if final_selections[key]:
#                 selection_mask[torch.tensor(final_selections[key], device=self.lsa_device)] = 1.0
#
#             # 🛠️ 修复：向左缩进，移出 if 块！无论是否投毒都必须生成 final_g_layer
#             # 带有 Gamma 放大的强力组装 (若 mask 全为 0，则自动回退为纯良性更新 g_b_flat)
#             final_g_layer = selection_mask * (g_p_flat * self.gamma) + (1.0 - selection_mask) * g_b_flat
#
#             p_delta = final_g_layer.reshape(*param_shapes[key]).cpu()
#             poisoned_state[key] = (global_model_state[key].float() + p_delta).to(local_model_state[key].dtype)
#
#         used_total = global_budget - remaining_budget
#         print(f"   [Sign-Aligned BC-DARE] 策略 B 执行完毕：耗尽预算 {used_total}/{global_budget}")
#         print(
#             f"   [Sign-Aligned BC-DARE] 预算利用率: {(used_total / global_budget) * 100:.2f}% (剩余 {remaining_budget} 为候选池物理枯竭)\n")
#
#         self.attack_invoke_count += 1
#         return poisoned_state
import torch
import math
from typing import Dict, Any

#
# 注意：请确保文件顶部已经导入了所需的基础库，并且继承了 BaseAttack
# class BaseAttack(...) 等定义保持不变

class FedDAREAttack(BaseAttack):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 0)
        # 严格执行全局 1% 预算 (drop_rate 0.99 意味着抛弃 99%，只保留 1%)
        self.drop_rate = config.get('drop_rate', 0.99)
        # 恢复 Gamma 放大，用于突破服务器聚合时的稀释效应
        self.gamma = config.get('gamma', 50.0)

        input_dim = config.get('input_dim', 32)
        default_size = 5 if input_dim >= 32 else 4
        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)

        # 缓存与 LSA 环境参数
        self.lsa_model = None
        self.lsa_dataloader = None
        self.lsa_device = None
        self.cached_layer_scores = None
        self.cached_score_sum = None

        # 动态 LSA 测算计数器
        self.attack_invoke_count = 0
        self.lsa_recompute_interval = 10  # 每 10 轮重测一次关键层，节省算力

    def get_data_type(self) -> str:
        return "image"

    def setup_lsa_environment(self, model: torch.nn.Module, dataloader: Any, device: torch.device):
        """挂载 LSA 测试环境，用于动态计算层级后门敏感度"""
        self.lsa_model = model
        self.lsa_dataloader = dataloader
        self.lsa_device = device

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                              poison_indices: torch.Tensor) -> None:
        """在图片右下角打上 5x5 的纯白 Trigger"""
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        row_start, row_end = h - self.trigger_height, h
        col_start, col_end = w - self.trigger_width, w
        # 将 trigger 区域设置为最大值 (标准化前通常对应白色)
        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = torch.max(poisoned_data)
        poisoned_labels[poison_indices] = self.target_class

    def _evaluate_bsr(self, eval_model: torch.nn.Module) -> float:
        """内部评估函数：测试当前模型权重的真实后门成功率 (BSR)"""
        eval_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.lsa_dataloader:
                data, target = data.to(self.lsa_device), target.to(self.lsa_device)

                # 只测试原本不是 target_class 的样本
                mask = (target != self.target_class)
                if not mask.any(): continue
                data = data[mask]

                poisoned_data = data.clone()
                poison_indices = torch.arange(data.shape[0])
                # 先反标准化，打上 trigger，再标准化回去
                poisoned_data = self._denormalize(poisoned_data)
                dummy_labels = torch.zeros(data.shape[0], dtype=torch.long, device=self.lsa_device)
                self._apply_static_trigger(poisoned_data, dummy_labels, poison_indices)
                poisoned_data = self._normalize(poisoned_data)

                outputs = eval_model(poisoned_data)
                _, predicted = torch.max(outputs.data, 1)
                total += data.size(0)
                correct += (predicted == self.target_class).sum().item()

        return correct / total if total > 0 else 0.0

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              benign_model_state: Dict[str, torch.Tensor] = None,
                              algorithm: str = "FedAvg") -> Dict[str, torch.Tensor]:
        if benign_model_state is None:
            raise ValueError("Sign-Aligned BC-DARE 必须依赖 benign_model_state 进行符号对齐与基准测算！")

        G_poison = {}
        G_benign = {}
        param_shapes = {}
        total_params = 0

        # 计算增量梯度 (Delta W)
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                continue
            G_poison[key] = local_model_state[key].float() - global_model_state[key].float()
            G_benign[key] = benign_model_state[key].float() - global_model_state[key].float()
            param_shapes[key] = G_poison[key].shape
            total_params += G_poison[key].numel()

            # =====================================================================
            # 🌟 新增埋点代码：记录未经放大的恶意梯度与良性梯度的 L2 范数
            # =====================================================================
            import math
            import os

            norm_poison_sq = sum(torch.sum(G_poison[k] ** 2).item() for k in G_poison)
            norm_benign_sq = sum(torch.sum(G_benign[k] ** 2).item() for k in G_benign)
            norm_p = math.sqrt(norm_poison_sq)
            norm_b = math.sqrt(norm_benign_sq)

            # 将数据追加写入 CSV 文件中，方便后续画图
            log_file = "gradient_norms_empirical_proof.csv"
            if not os.path.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("Round,Poison_Norm,Benign_Norm\n")

            with open(log_file, "a") as f:
                f.write(f"{self.attack_invoke_count},{norm_p:.6f},{norm_b:.6f}\n")

        # =========================================================================
        # 第一阶段：层级敏感度分析 (Layer-wise Sensitivity Analysis)
        # =========================================================================
        force_recompute = (self.attack_invoke_count % self.lsa_recompute_interval == 0)

        if self.cached_layer_scores is None or force_recompute:
            print(f"\n   [Sign-Aligned BC-DARE] {'首次' if not force_recompute else '周期性'}执行严格 LSA 打分...")
            layer_scores = {}
            score_sum = 0.0
            self.lsa_model = self.lsa_model.to(self.lsa_device)

            # 测试纯毒模型的基准 BSR
            self.lsa_model.load_state_dict(local_model_state)
            bsr_malicious = self._evaluate_bsr(self.lsa_model)

            for key in G_poison.keys():
                # 逐层替换为良性参数
                hybrid_state = {k: v.clone() for k, v in local_model_state.items()}
                hybrid_state[key] = benign_model_state[key].clone()
                self.lsa_model.load_state_dict(hybrid_state)
                bsr_hybrid = self._evaluate_bsr(self.lsa_model)

                # 计算杀伤力下降幅度
                delta_bsr = bsr_malicious - bsr_hybrid
                # 严格非负：使用微小 epsilon (1e-9) 确保即使是 0 分层也有极小的入选几率，防止死锁
                score = max(1e-9, delta_bsr)
                layer_scores[key] = score
                score_sum += score

            self.cached_layer_scores = layer_scores
            self.cached_score_sum = score_sum
            self.lsa_model = self.lsa_model.cpu()
        else:
            layer_scores = self.cached_layer_scores
            score_sum = self.cached_score_sum

        # =========================================================================
        # 第二阶段：符号对齐 (Sign-Alignment) 与 水箱分配 (Water-filling)
        # =========================================================================
        global_budget = int(total_params * (1.0 - self.drop_rate))
        remaining_budget = global_budget

        layer_candidates = {}
        for key in G_poison.keys():
            g_p_flat = G_poison[key].flatten().to(self.lsa_device)
            g_b_flat = G_benign[key].flatten().to(self.lsa_device)

            # 核心防御绕过：仅保留符号与良性更新严格一致的参数维度
            sign_mask = (torch.sign(g_p_flat) == torch.sign(g_b_flat))
            candidate_indices = torch.where(sign_mask)[0]

            if len(candidate_indices) > 0:
                epsilon = 1e-8
                # 计算显著性：恶意的扰动有多强
                saliency = (torch.abs(g_p_flat[candidate_indices]) / (torch.abs(g_b_flat[candidate_indices]) + epsilon))
                sorted_inner = torch.argsort(saliency, descending=True)
                layer_candidates[key] = candidate_indices[sorted_inner].tolist()
            else:
                layer_candidates[key] = []

        # 迭代式分配预算
        final_selections = {key: [] for key in G_poison.keys()}
        active_layers = [k for k in layer_scores.keys() if len(layer_candidates[k]) > 0]

        while remaining_budget > 0 and len(active_layers) > 0:
            current_total_score = sum(layer_scores[k] for k in active_layers)
            layers_to_remove = []
            budget_to_distribute = remaining_budget

            for key in active_layers:
                # 按照 LSA 得分比例分配名额
                allocation = int(math.floor(budget_to_distribute * (layer_scores[key] / current_total_score)))
                if allocation == 0 and remaining_budget > 0: allocation = 1

                take = min(allocation, len(layer_candidates[key]), remaining_budget)

                if take > 0:
                    final_selections[key].extend(layer_candidates[key][:take])
                    layer_candidates[key] = layer_candidates[key][take:]
                    remaining_budget -= take

                if len(layer_candidates[key]) == 0:
                    layers_to_remove.append(key)
                if remaining_budget <= 0: break

            for k in layers_to_remove: active_layers.remove(k)

        # =========================================================================
        # 第三阶段：Gamma 能量重组
        # =========================================================================
        poisoned_state = {}
        for key in local_model_state.keys():
            if key not in G_poison:
                poisoned_state[key] = benign_model_state[key].clone()
                continue

            g_p_flat = G_poison[key].flatten().to(self.lsa_device)
            g_b_flat = G_benign[key].flatten().to(self.lsa_device)

            selection_mask = torch.zeros_like(g_p_flat)
            if final_selections[key]:
                selection_mask[torch.tensor(final_selections[key], device=self.lsa_device)] = 1.0

            # 核心融合：被选中的 1% 放大 Gamma 倍，剩下的 99% 使用良性参数掩盖
            final_g_layer = selection_mask * (g_p_flat * self.gamma) + (1.0 - selection_mask) * g_b_flat

            p_delta = final_g_layer.reshape(*param_shapes[key]).cpu()
            poisoned_state[key] = (global_model_state[key].float() + p_delta).to(local_model_state[key].dtype)

        used_total = global_budget - remaining_budget
        print(
            f"   [Sign-Aligned BC-DARE] 预算利用率: {(used_total / global_budget) * 100:.2f}% (Gamma 放大: {self.gamma}x)")

        # =========================================================================
        # 第四阶段：全局范数投影 (Global Norm Projection / PGD 核心)
        # 数学闭环：计算并施加投影因子 ρ，严格控制全局 L2 范数，绕过 NormClipping
        # =========================================================================
        total_norm_crafted_sq = 0.0
        total_norm_benign_sq = 0.0

        # 1. 遍历所有层，计算整体攻击梯度与良性梯度的 L2 范数的平方和
        for key in G_poison.keys():
            g_c = poisoned_state[key].float() - global_model_state[key].float()
            g_b = benign_model_state[key].float() - global_model_state[key].float()

            total_norm_crafted_sq += torch.sum(g_c ** 2).item()
            total_norm_benign_sq += torch.sum(g_b ** 2).item()

        norm_crafted = math.sqrt(total_norm_crafted_sq)
        norm_benign = math.sqrt(total_norm_benign_sq)

        # 2. 如果攻击范数超标，则计算投影因子 ρ 进行等比例压缩
        if norm_crafted > norm_benign and norm_crafted > 1e-9:
            rho = norm_benign / norm_crafted  # 这就是您数学推导中的 ρ
            print(
                f"   [PGD 投影触发] 恶意范数 {norm_crafted:.4f} > 良性范数 {norm_benign:.4f} | 投影因子 ρ = {rho:.4f}")

            for key in G_poison.keys():
                g_c = poisoned_state[key].float() - global_model_state[key].float()
                # 施加等比例投影缩放
                projected_delta = g_c * rho
                # 写回最终模型状态
                poisoned_state[key] = (global_model_state[key].float() + projected_delta).to(
                    local_model_state[key].dtype)
        else:
            print(f"   [PGD 投影未触发] 恶意范数 {norm_crafted:.4f} 安全 (<= {norm_benign:.4f})")

        self.attack_invoke_count += 1
        return poisoned_state

class LayerwisePoisoningAttack(BadNetsAttack):
    """
    严格复现 ICLR 2024: BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS (LP Attack)
    基于官方开源仓库 models/Attacker.py 中的逻辑重写
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 对应官方选项 --cifar_scale (1.0) / --cnn_scale (0.5)，控制隐身与放大
        self.lambda_val = config.get('lambda_val', 1.0)
        # 对应官方选项 --mode10_tau (0.95)，控制 m2b 的恢复阈值
        self.tau = config.get('tau', 0.95)

        self.lsa_model = None
        self.lsa_dataloader = None
        self.lsa_device = None
        self.cached_bc_layers = None

        self.attack_invoke_count = 0
        # 为了加速联邦学习实验，每隔若干轮重新测算一次 BC 层
        self.lsa_recompute_interval = config.get('lsa_recompute_interval', 10)

    def setup_lsa_environment(self, model: torch.nn.Module, dataloader: Any, device: torch.device):
        """由 client.py 在每轮训练前调用，挂载 LSA 环境"""
        self.lsa_model = model
        self.lsa_dataloader = dataloader
        self.lsa_device = device

    def _evaluate_bsr(self, eval_model: torch.nn.Module) -> float:
        """评估给定模型上的后门成功率 (BSR)"""
        correct, total = 0, 0
        eval_model.eval()
        with torch.no_grad():
            for data, target in self.lsa_dataloader:
                data = data.to(self.lsa_device)
                poisoned_data = data.clone()
                poison_indices = torch.arange(data.shape[0])

                poisoned_data = self._denormalize(poisoned_data)
                dummy_labels = torch.zeros(data.shape[0], dtype=torch.long, device=self.lsa_device)
                # 动态打上 Trigger，且将 Label 设为 Target Class
                self._apply_static_trigger(poisoned_data, dummy_labels, poison_indices)
                poisoned_data = self._normalize(poisoned_data)

                outputs = eval_model(poisoned_data)
                _, predicted = torch.max(outputs.data, 1)

                total += data.size(0)
                correct += (predicted == self.target_class).sum().item()
        return correct / total if total > 0 else 0.0

    def _run_lsa_and_find_bc_layers(self, benign_state_proxy: Dict[str, torch.Tensor],
                                    malicious_state: Dict[str, torch.Tensor]) -> List[str]:
        """严格执行官方代码的 FLS (Forward) 和 BLS (Backward) 层替换分析"""
        if self.lsa_model is None or self.lsa_dataloader is None:
            raise ValueError("LSA Environment not setup! Make sure client.py calls setup_lsa_environment.")

        self.lsa_model = self.lsa_model.to(self.lsa_device)
        print(f"\n   [{self.name}] 严格执行官方 LP b2m / m2b 分析流程...")

        # 1. 测算满毒模型基准 BSR (对应官方 FLS 函数准备阶段)
        self.lsa_model.load_state_dict(malicious_state)
        bsr_malicious = self._evaluate_bsr(self.lsa_model)
        print(f"   [{self.name}] 基准恶意模型 BSR: {bsr_malicious:.4f}")

        if bsr_malicious == 0:
            print(f"   [{self.name}] 警告：恶意模型 BSR 为 0，降级为全层攻击。")
            return list(malicious_state.keys())

        # 2. b2m 前向替换打分 (对应官方 FLS 函数中的循环)
        layer_delta_bsr = {}
        for layer_name in malicious_state.keys():
            if 'num_batches_tracked' in layer_name or 'running_mean' in layer_name or 'running_var' in layer_name:
                continue

            # Hybrid 模型：主体是恶意的，唯独把当前层换成纯净的
            hybrid_state = {k: v.clone() for k, v in malicious_state.items()}
            hybrid_state[layer_name] = benign_state_proxy[layer_name].clone()

            self.lsa_model.load_state_dict(hybrid_state)
            bsr_hybrid = self._evaluate_bsr(self.lsa_model)

            # 计算因果破坏力 (Delta BSR)
            delta_bsr = bsr_malicious - bsr_hybrid
            layer_delta_bsr[layer_name] = max(0.0, delta_bsr)

            if delta_bsr > 0:
                print(f"   [{self.name}] b2m 替换层 {layer_name:<25} | Delta BSR: {delta_bsr:.4f}")

        # 按照破坏力降序排列
        sorted_layers = sorted(layer_delta_bsr.items(), key=lambda x: x[1], reverse=True)

        # 3. m2b 后向替换验证 (对应官方 BLS_weight 函数)
        print(f"   [{self.name}] 执行 m2b 后向验证, 寻找满足 tau={self.tau} 的最小 BC 集合...")
        target_bsr = self.tau * bsr_malicious
        bc_layers_found = []

        # 从全纯净模型起步
        m2b_state = {k: v.clone() for k, v in benign_state_proxy.items()}

        for layer_name, score in sorted_layers:
            # 依次将破坏力最强的恶意层塞回去
            m2b_state[layer_name] = malicious_state[layer_name].clone()
            bc_layers_found.append(layer_name)

            self.lsa_model.load_state_dict(m2b_state)
            current_m2b_bsr = self._evaluate_bsr(self.lsa_model)

            if current_m2b_bsr >= target_bsr:
                print(
                    f"   [{self.name}] m2b 锁定完成! 命中 {len(bc_layers_found)} 个 BC 层 (当前 BSR: {current_m2b_bsr:.4f}).")
                break

        # 还原设备环境
        self.lsa_model.load_state_dict(malicious_state)
        self.lsa_model = self.lsa_model.cpu()
        return bc_layers_found

    def apply_model_poisoning(self,
                              local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              benign_model_state: Dict[str, torch.Tensor] = None,
                              algorithm: str = 'FedAvg',
                              **kwargs) -> Dict[str, torch.Tensor]:

        # 官方逻辑的核心：必须存在 benign_w (纯净代理) 才能实现完美的隐身
        if benign_model_state is None:
            print(f"   [{self.name}] 警告：未检测到 benign_model_state，降级使用 global_model_state 作为良性代理。")
            benign_model_state = global_model_state

        force_recompute = (self.attack_invoke_count % self.lsa_recompute_interval == 0)

        if self.cached_bc_layers is None or force_recompute:
            dynamic_bc_layers = self._run_lsa_and_find_bc_layers(benign_model_state, local_model_state)
            self.cached_bc_layers = dynamic_bc_layers
        else:
            rounds_left = self.lsa_recompute_interval - (self.attack_invoke_count % self.lsa_recompute_interval)
            print(
                f"   [{self.name}] 重用已缓存的 {len(self.cached_bc_layers)} 个 BC 层 (距下次重测还有 {rounds_left} 轮)。")
            dynamic_bc_layers = self.cached_bc_layers

        self.attack_invoke_count += 1

        poisoned_state = {}
        bc_param_count = 0
        total_param_count = 0

        # =====================================================================
        # 严格复刻官方 Github 中 models/Attacker.py 的 craft_model()
        # =====================================================================
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                poisoned_state[key] = local_model_state[key].clone()
                continue

            total_param_count += local_model_state[key].numel()
            is_bc_layer = key in dynamic_bc_layers

            if is_bc_layer:
                # 官方公式: crafted = global + lambda*(malicious - global) + max(0, 1-lambda)*(benign - global)
                # 实现了: lambda <= 1 时为平滑插值，lambda > 1 时转化为暴力 Scaling Attack
                delta_m = local_model_state[key].float() - global_model_state[key].float()
                delta_b = benign_model_state[key].float() - global_model_state[key].float()

                poisoned_delta = (self.lambda_val * delta_m) + (max(0.0, 1.0 - self.lambda_val) * delta_b)
                poisoned_state[key] = (global_model_state[key].float() + poisoned_delta).to(
                    local_model_state[key].dtype)
                bc_param_count += local_model_state[key].numel()
            else:
                # 官方隐身灵魂：对于非 BC 层，并非抛弃，而是上传该客户端训练出来的纯净参数 (benign_model_state)
                # 这极大地伪装了 L2 距离的分布。
                poisoned_state[key] = benign_model_state[key].clone()

        if total_param_count > 0:
            print(
                f"   [{self.name}] 官方 LP 组装完毕: 毒化 {len(dynamic_bc_layers)} 个层 (占参数 {(bc_param_count / total_param_count) * 100:.2f}%)，Lambda = {self.lambda_val}\n")

        return poisoned_state


class MinMaxAttack:
    def __init__(self, dev_type='std'):
        self.dev_type = dev_type

    def _flatten_weights(self, weights_dict):
        return torch.cat([v.flatten() for v in weights_dict.values()])

    def _unflatten_weights(self, flat_tensor, template_dict):
        unflat_dict = {}
        idx = 0
        for k, v in template_dict.items():
            numel = v.numel()
            unflat_dict[k] = flat_tensor[idx:idx + numel].view_as(v).clone()
            idx += numel
        return unflat_dict

    def apply_attack(self, malicious_local_updates):
        if len(malicious_local_updates) == 1:
            return malicious_local_updates[0]

        flat_updates = [self._flatten_weights(w) for w in malicious_local_updates]
        stacked_updates = torch.stack(flat_updates)

        mu = torch.mean(stacked_updates, dim=0)

        if self.dev_type == 'sign':
            deviation = -torch.sign(mu)
        elif self.dev_type == 'std':
            deviation = -torch.std(stacked_updates, dim=0, unbiased=False)
        elif self.dev_type == 'unit':
            deviation = -mu / (torch.norm(mu) + 1e-8)
        else:
            deviation = -torch.std(stacked_updates, dim=0, unbiased=False)

        distances = torch.cdist(stacked_updates, stacked_updates, p=2.0)
        max_distance = torch.max(distances)

        gamma_succ = 0.0
        gamma_fail = 100.0
        gamma = gamma_fail / 2.0
        threshold = 1e-4

        while abs(gamma_fail - gamma_succ) > threshold:
            candidate_malicious = mu + gamma * deviation
            dist_to_locals = torch.norm(stacked_updates - candidate_malicious, dim=1)

            if torch.max(dist_to_locals) <= max_distance:
                gamma_succ = gamma
                gamma = gamma + (gamma_fail - gamma) / 2.0
            else:
                gamma_fail = gamma
                gamma = gamma - (gamma - gamma_succ) / 2.0

        best_malicious_flat = mu + gamma_succ * deviation
        template = malicious_local_updates[0]
        malicious_state_dict = self._unflatten_weights(best_malicious_flat, template)

        return malicious_state_dict


class TrimAttack:
    def __init__(self, num_attackers=1, num_total_clients=10, b=2.0):
        self.num_attackers = num_attackers
        self.num_total = num_total_clients
        self.b = b

    def _flatten_weights(self, weights_dict):
        return torch.cat([v.flatten() for v in weights_dict.values()])

    def _unflatten_weights(self, flat_tensor, template_dict):
        unflat_dict = {}
        idx = 0
        for k, v in template_dict.items():
            numel = v.numel()
            unflat_dict[k] = flat_tensor[idx:idx + numel].view_as(v).clone()
            idx += numel
        return unflat_dict

    def apply_attack(self, malicious_local_updates):
        if len(malicious_local_updates) == 1:
            return malicious_local_updates[0]

        flat_updates = torch.stack([self._flatten_weights(w) for w in malicious_local_updates])
        mu = torch.mean(flat_updates, dim=0)
        std = torch.std(flat_updates, dim=0, unbiased=False) + 1e-9

        direction = torch.sign(mu)
        best_malicious_flat = mu - self.b * std * direction

        return self._unflatten_weights(best_malicious_flat, malicious_local_updates[0])


class KrumAttack:
    def __init__(self, num_attackers=1, num_total_clients=10):
        self.num_attackers = num_attackers
        self.num_total = num_total_clients

    def _flatten_weights(self, weights_dict):
        return torch.cat([v.flatten() for v in weights_dict.values()])

    def _unflatten_weights(self, flat_tensor, template_dict):
        unflat_dict = {}
        idx = 0
        for k, v in template_dict.items():
            numel = v.numel()
            unflat_dict[k] = flat_tensor[idx:idx + numel].view_as(v).clone()
            idx += numel
        return unflat_dict

    def apply_attack(self, malicious_local_updates):
        if len(malicious_local_updates) == 1:
            return malicious_local_updates[0]

        flat_updates = torch.stack([self._flatten_weights(w) for w in malicious_local_updates])
        mu = torch.mean(flat_updates, dim=0)
        std = torch.std(flat_updates, dim=0, unbiased=False) + 1e-9
        direction = torch.sign(mu)

        gamma_succ = 0.0
        gamma_fail = 3.0
        gamma = gamma_fail / 2.0
        threshold = 1e-4

        krum_k = max(1, self.num_total - self.num_attackers - 2)

        def calc_krum_score(target_vec, pool_benign, cand_vec, num_mal):
            dists_to_benign = torch.cdist(target_vec.unsqueeze(0), pool_benign)[0]
            dists_to_mal = torch.cdist(target_vec.unsqueeze(0), cand_vec.unsqueeze(0))[0].repeat(num_mal)
            all_dists = torch.cat([dists_to_benign, dists_to_mal])
            all_dists, _ = torch.sort(all_dists)
            return torch.sum(all_dists[1:krum_k + 1])

        while abs(gamma_fail - gamma_succ) > threshold:
            candidate = mu - gamma * std * direction

            score_candidate = calc_krum_score(candidate, flat_updates, candidate, self.num_attackers)
            score_benign = calc_krum_score(mu, flat_updates, candidate, self.num_attackers)

            if score_candidate < score_benign:
                gamma_succ = gamma
                gamma = gamma + (gamma_fail - gamma) / 2.0
            else:
                gamma_fail = gamma
                gamma = gamma - (gamma - gamma_succ) / 2.0

        best_malicious_flat = mu - gamma_succ * std * direction
        return self._unflatten_weights(best_malicious_flat, malicious_local_updates[0])


class CerPAttack:
    def __init__(self, config):
        self.config = config
        self.name = 'CerPAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.adversarial_clients = config.get('apply_to_client_ids', [])

        if self.client_id in self.adversarial_clients:
            self.attacker_idx = self.adversarial_clients.index(self.client_id)
        else:
            self.attacker_idx = 0

        self.num_attackers = max(1, len(self.adversarial_clients))
        self.trigger_size = config.get('trigger_width', 4)
        self.epsilon = config.get('epsilon', 0.5)

    def use_model_trigger(self):
        return False

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def poison_data(self, data, target):
        poisoned_data = data.clone()
        poisoned_target = target.clone()

        batch_size = data.size(0)
        channels, height, width = data.size()[1:]

        start_h = height - self.trigger_size
        start_w = width - self.trigger_size

        chunk_size = max(1, self.trigger_size // self.num_attackers)
        my_chunk_start = self.attacker_idx * chunk_size
        my_chunk_end = my_chunk_start + chunk_size if self.attacker_idx < self.num_attackers - 1 else self.trigger_size

        poison_ratio = self.config.get('poison_ratio', 0.5)
        num_poisoned = int(batch_size * poison_ratio)

        if num_poisoned > 0:
            poisoned_target[:num_poisoned] = self.target_class

            for i in range(num_poisoned):
                for c in range(channels):
                    poisoned_data[i, c,
                    start_h + my_chunk_start: start_h + my_chunk_end,
                    start_w: start_w + self.trigger_size] = 2.5

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        poisoned_state = {}
        update_vector = []
        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            update_vector.append(diff.view(-1))

        update_vector = torch.cat(update_vector)
        l2_norm = torch.norm(update_vector, p=2)

        if l2_norm > self.epsilon:
            scale_factor = self.epsilon / (l2_norm + 1e-9)
            print(f"   [CerP] Client {self.client_id} clipping model update (Norm {l2_norm:.2f} -> {self.epsilon})")

            for k in local_model_state.keys():
                diff = local_model_state[k] - global_model_state[k]
                poisoned_state[k] = global_model_state[k] + diff * scale_factor
        else:
            poisoned_state = copy.deepcopy(local_model_state)

        return poisoned_state


class A3FLAttack:
    def __init__(self, config):
        self.config = config
        self.name = 'A3FLAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])

        self.trigger_height = config.get('trigger_height', 5)
        self.trigger_width = config.get('trigger_width', 5)
        self.trigger_patch = None

    def use_model_trigger(self):
        return False

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def train_attack_model(self, model, dataloader, client_id, device, verbose=False):
        if client_id not in self.apply_to_client_ids:
            return

        print(f"   [A3FL] Optimizing Adversarially Adaptive Trigger for client {client_id}...")

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if self.trigger_patch is None:
            self.trigger_patch = torch.zeros((3, self.trigger_height, self.trigger_width), device=device,
                                             requires_grad=True)
        else:
            self.trigger_patch = self.trigger_patch.detach().clone().to(device)
            self.trigger_patch.requires_grad = True

        optimizer_trigger = torch.optim.Adam([self.trigger_patch], lr=0.01)
        criterion = nn.CrossEntropyLoss()

        epochs = self.config.get('adv_epochs', 3)
        for epoch in range(epochs):
            for data, target in dataloader:
                data = data.to(device)

                batch_size = data.size(0)
                poison_ratio = self.config.get('poison_ratio', 0.5)
                num_poison = int(batch_size * poison_ratio)
                if num_poison == 0: continue

                poisoned_data = data[:num_poison].clone()
                channels, h, w = poisoned_data.shape[1:]
                start_h, start_w = h - self.trigger_height, w - self.trigger_width

                poisoned_data[:, :, start_h:, start_w:] = poisoned_data[:, :, start_h:, start_w:] + self.trigger_patch
                poisoned_target = torch.ones(num_poison, dtype=torch.long, device=device) * self.target_class

                optimizer_trigger.zero_grad()
                output = model(poisoned_data)

                loss = criterion(output, poisoned_target)
                loss.backward()
                optimizer_trigger.step()

                with torch.no_grad():
                    self.trigger_patch.clamp_(-0.5, 0.5)

        self.trigger_patch.requires_grad = False

        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def poison_data(self, data, target):
        poisoned_data = data.clone()
        poisoned_target = target.clone()

        batch_size = data.size(0)
        poison_ratio = self.config.get('poison_ratio', 0.5)
        num_poisoned = int(batch_size * poison_ratio)

        if num_poisoned > 0 and self.trigger_patch is not None:
            poisoned_target[:num_poisoned] = self.target_class

            channels, h, w = poisoned_data.shape[1:]
            start_h, start_w = h - self.trigger_height, w - self.trigger_width

            device = poisoned_data.device
            optimized_patch = self.trigger_patch.to(device)

            poisoned_data[:num_poisoned, :, start_h:, start_w:] = poisoned_data[:num_poisoned, :, start_h:,
                                                                  start_w:] + optimized_patch

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        scaling_factor = self.config.get('scaling_factor', 1)
        if scaling_factor > 1:
            poisoned_state = {}
            for k in local_model_state.keys():
                diff = local_model_state[k] - global_model_state[k]
                poisoned_state[k] = global_model_state[k] + diff * scaling_factor
            return poisoned_state
        return local_model_state


class FCBAAttack:
    def __init__(self, config):
        self.config = config
        self.name = 'FCBAAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])

        self.trigger_size = config.get('trigger_width', 4)
        self.scaling_factor = config.get('scaling_factor', 1)

    def use_model_trigger(self):
        return False

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def poison_data(self, data, target):
        poisoned_data = data.clone()
        poisoned_target = target.clone()

        batch_size = data.size(0)
        channels, height, width = data.size()[1:]

        poison_ratio = self.config.get('poison_ratio', 0.5)
        num_poisoned = int(batch_size * poison_ratio)

        if num_poisoned > 0:
            poisoned_target[:num_poisoned] = self.target_class
            pixel_val = 2.5

            poisoned_data[:num_poisoned, :, 0:self.trigger_size, 0:self.trigger_size] = pixel_val
            poisoned_data[:num_poisoned, :, 0:self.trigger_size, width - self.trigger_size:width] = pixel_val
            poisoned_data[:num_poisoned, :, height - self.trigger_size:height, 0:self.trigger_size] = pixel_val
            poisoned_data[:num_poisoned, :, height - self.trigger_size:height,
            width - self.trigger_size:width] = pixel_val

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        if self.scaling_factor <= 1:
            return local_model_state

        poisoned_state = {}
        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            poisoned_state[k] = global_model_state[k] + diff * self.scaling_factor

        return poisoned_state


class IBAGenerator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class IBAAttack:
    def __init__(self, config):
        self.config = config
        self.name = 'IBAAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])

        self.epsilon = config.get('epsilon', 0.1)
        self.pgd_bound = config.get('pgd_bound', 2.0)
        self.scaling_factor = config.get('scaling_factor', 1)

        self.generator = None

    def use_model_trigger(self):
        return False

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def train_attack_model(self, model, dataloader, client_id, device, verbose=False):
        if client_id not in self.apply_to_client_ids:
            return

        print(f"   [IBA] Training instance-specific trigger generator for client {client_id}...")

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if self.generator is None:
            sample_data, _ = next(iter(dataloader))
            channels = sample_data.shape[1]
            self.generator = IBAGenerator(channels).to(device)

        self.generator.train()
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        epochs = self.config.get('adv_epochs', 2)
        for epoch in range(epochs):
            for data, target in dataloader:
                data = data.to(device)
                batch_size = data.size(0)

                noise = self.generator(data) * self.epsilon
                poisoned_data = data + noise

                poisoned_target = torch.ones(batch_size, dtype=torch.long, device=device) * self.target_class

                optimizer_g.zero_grad()
                output = model(poisoned_data)
                loss = criterion(output, poisoned_target)
                loss.backward()
                optimizer_g.step()

        self.generator.eval()

        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def poison_data(self, data, target):
        poisoned_data = data.clone()
        poisoned_target = target.clone()

        if self.generator is None:
            return poisoned_data, poisoned_target

        batch_size = data.size(0)
        poison_ratio = self.config.get('poison_ratio', 0.5)
        num_poisoned = int(batch_size * poison_ratio)

        if num_poisoned > 0:
            poisoned_target[:num_poisoned] = self.target_class

            with torch.no_grad():
                device = poisoned_data.device
                clean_subset = poisoned_data[:num_poisoned].to(device)

                noise = self.generator(clean_subset) * self.epsilon
                poisoned_subset = clean_subset + noise
                poisoned_data[:num_poisoned] = poisoned_subset

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        poisoned_state = {}

        update_vector = []
        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            update_vector.append(diff.view(-1))

        update_vector = torch.cat(update_vector)
        l2_norm = torch.norm(update_vector, p=2)

        scale_factor = 1.0
        if l2_norm > self.pgd_bound:
            scale_factor = self.pgd_bound / (l2_norm + 1e-9)
            print(f"   [IBA] Applying PGD constraint: L2-Norm clipped from {l2_norm:.2f} down to {self.pgd_bound}")

        final_scale = scale_factor * self.scaling_factor

        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            poisoned_state[k] = global_model_state[k] + diff * final_scale

        return poisoned_state


class DarkFedAttack(BaseAttack):
    """
    DarkFed: A Data-Free Backdoor Attack in Federated Learning (IJCAI 2024)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 0)
        self.trigger_size = config.get('trigger_width', 5)

        self.safe_norm_bound = config.get('safe_norm_bound', 2.0)

    def use_model_trigger(self):
        return False

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                              poison_indices: torch.Tensor) -> None:
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]

        for idx in poison_indices:
            noise = torch.rand_like(poisoned_data[idx])
            poisoned_data[idx] = noise

            row_start = h - self.trigger_size
            row_end = h
            col_start = w - self.trigger_size
            col_end = w

            poisoned_data[idx, :, row_start:row_end, col_start:col_end] = 1.0

            poisoned_labels[idx] = self.target_class

    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        poisoned_state = {}

        update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                continue
            update_dict[key] = (local_model_state[key].cpu() - global_model_state[key].cpu()).float()

        vec_list = [v.numpy().flatten() for v in update_dict.values()]
        if not vec_list:
            return local_model_state

        import numpy as np
        update_vec = np.concatenate(vec_list)
        update_vec = np.nan_to_num(update_vec, nan=0.0, posinf=0.0, neginf=0.0)
        current_norm = np.linalg.norm(update_vec)

        if current_norm == 0:
            return local_model_state

        scale_factor = 1.0
        if current_norm > self.safe_norm_bound:
            scale_factor = self.safe_norm_bound / current_norm

        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                    poisoned_state[key] = local_model_state[key].clone()
                    continue

                mimic_delta = update_dict[key] * scale_factor
                poisoned_state[key] = (global_model_state[key].cpu().float() + mimic_delta).to(
                    local_model_state[key].dtype)

        return poisoned_state

def create_attack(attack_config: Dict[str, Any]) -> BaseAttack:
    """Factory function to create attack instances"""
    attack_name = attack_config['name']

    if attack_name == 'BadNetsAttack':
        return BadNetsAttack(attack_config)
    elif attack_name == 'BlendedAttack':
        return BlendedAttack(attack_config)
    elif attack_name == 'DBAAttack':
        return DBAAttack(attack_config)
    elif attack_name == 'SinusoidalAttack':
        return SinusoidalAttack(attack_config)
    elif attack_name == 'LabelFlippingAttack':
        return LabelFlippingAttack(attack_config)
    elif attack_name == 'ModelReplacementAttack':
        return ModelReplacementAttack(attack_config)
    elif attack_name == 'NeurotoxinAttack':
        return NeurotoxinAttack(attack_config)
    elif attack_name == 'EdgeCaseBackdoorAttack':
        return EdgeCaseBackdoorAttack(attack_config)
    elif attack_name == 'ThreeDFedAttack':
        return ThreeDFedAttack(attack_config)
    elif attack_name == 'FedDAREAttack':
        return FedDAREAttack(attack_config)
    elif attack_name == 'LayerwisePoisoningAttack':
        return LayerwisePoisoningAttack(attack_config)
    elif attack_name in ['MinMaxAttack', 'TrimAttack', 'KrumAttack']:
        class DummyCollusionAttack(BaseAttack):
            def get_data_type(self) -> str:
                return "image"

            def _apply_static_trigger(self, *args, **kwargs):
                pass

            def apply_model_poisoning(self, local_model_state, global_model_state, algorithm, **kwargs):
                return local_model_state

        return DummyCollusionAttack(attack_config)
    elif attack_name == 'CerPAttack':
        return CerPAttack(attack_config)
    elif attack_name == 'A3FLAttack':
        return A3FLAttack(attack_config)
    elif attack_name == 'FCBAAttack':
        return FCBAAttack(attack_config)
    elif attack_name == 'IBAAttack':
        return IBAAttack(attack_config)
    elif attack_name == 'DarkFedAttack':
        return DarkFedAttack(attack_config)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")