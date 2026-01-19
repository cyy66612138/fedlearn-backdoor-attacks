"""
Attack implementations for federated learning
"""

from numpy.random import poisson
import torch
import torchvision
import os
import numpy as np
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pickle

def dump_poisoned_images(clean_imgs: torch.Tensor, clean_labels: torch.Tensor, 
                        poison_indices: torch.Tensor,
                        poisoned_imgs: torch.Tensor, poisoned_labels: torch.Tensor, 
                        noise_imgs: torch.Tensor, 
                        attack_config: Dict[str, Any]) -> None:
    """Dump poisoned images visualization to PNG file with labels"""
    # poison_indices = batch_size * poison_ratio
    # Make sure all tensors are in [0,1] for visualization
    def norm_img(x):
        return torch.clamp(x, 0.0, 1.0)

    clean_imgs = norm_img(clean_imgs)
    poisoned_imgs = norm_img(poisoned_imgs)
    # Use the pre-computed noise_imgs if provided, otherwise compute it
    noise_imgs = norm_img(noise_imgs)
    # Normalize noise for better visualization
    noise_imgs = norm_img((noise_imgs - noise_imgs.min()) / (noise_imgs.max() - noise_imgs.min() + 1e-8))

    dump_path = attack_config['dump_path']
    os.makedirs(dump_path, exist_ok=True)
    
    # Only take up to 9 images for each row
    n_show = min(9, clean_imgs.size(0))
    
    # Ensure all tensors are on the same device (CPU for visualization)
    clean_imgs = clean_imgs.cpu()
    poisoned_imgs = poisoned_imgs.cpu()
    noise_imgs = noise_imgs.cpu()
    
    # missing dataset name, client_id, round_idx, batch_idx
    dataset_name = attack_config['dataset_name']
    attack_name = attack_config['name']
    
    eps = attack_config.get('atk_eps', -1) # -1 means no perturbation attack

    # Add labels as text overlay if provided
    if clean_labels is not None and poisoned_labels is not None:
        
        # Convert tensors to numpy arrays
        clean_np = clean_imgs.permute(0, 2, 3, 1).cpu().numpy()
        poisoned_np = poisoned_imgs.permute(0, 2, 3, 1).cpu().numpy()
        noise_np = noise_imgs.permute(0, 2, 3, 1).cpu().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, n_show, figsize=(n_show * 2, 6))
        if n_show == 1:
            axes = axes.reshape(3, 1)
        
        row_labels = ['Clean Images', 'Poisoned Images', 'Residue']
        
        for i in range(3):  # 3 rows
            for j in range(n_show):  # n_show columns
                ax = axes[i, j]
                # Select the appropriate image data
                if i == 0:  # Clean images
                    img_data = clean_np[j]
                    label_text = f"{clean_labels[j].item()}"
                    label_color = 'blue'
                elif i == 1:  # Poisoned images
                    img_data = poisoned_np[j]
                    label_text = f"{poisoned_labels[j].item()}"
                    label_color = 'red'
                elif i == 2:  # Noise/Trigger
                    img_data = noise_np[j]
                    label_text = ""
                    label_color = 'black'
                
                # Display image with appropriate colormap
                if img_data.shape[-1] == 1:  # Grayscale image (MNIST, FashionMNIST)
                    ax.imshow(img_data.squeeze(-1), cmap='gray')
                else:  # Color image (CIFAR, etc.)
                    ax.imshow(img_data)
                ax.axis('off')
                
                # Add label if needed, adjust position based on label length
                if label_text:
                    label_len = len(label_text)
                    if label_len == 1:
                        xpos = 0.88
                    elif label_len == 2:
                        xpos = 0.80
                    else:  # 3 or more digits
                        xpos = 0.72
                else:
                    xpos = 0.88  # default, though label_text is empty
                if label_text:
                    ax.text(xpos, 0.96, label_text, transform=ax.transAxes, 
                           fontsize=14, fontweight='bold', color=label_color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                   edgecolor=label_color, alpha=0.8),
                           verticalalignment='top', horizontalalignment='left')
            
            # Add row label
            axes[i, 0].text(-0.15, 0.5, row_labels[i], transform=axes[i, 0].transAxes,
                           fontsize=12, fontweight='bold', rotation=90,
                           verticalalignment='center', horizontalalignment='center')
        
        plt.tight_layout()
        save_path = os.path.join(dump_path, f'{dataset_name}_{attack_name}_with_labels_{attack_config.get("target_class", -1)}_eps_{eps}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ Saved poisoned images with labels (3-row grid) to {save_path}")
        plt.close()
    # save all images tp .pickle file
    target_class = attack_config['target_class'] # -1 means all classes
    eps = attack_config.get('atk_eps', -1) # -1 means no perturbation attack
    latent_dim = attack_config.get('atk_latent_dim', -1)  # -1 means no latent dimension
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
        """Template method for poisoning data
        clean_data: [-2..., 2...] # normalized data
        clean_labels: [0, 1, 2, ..., 9] # original labels
        """
        # np.random.seed(self.seed) # remove this as it will cause the poison_indices to be different in each running time
        # np.random.seed()
        batch_size = clean_data.shape[0]
        client_id = self.config.get('client_id', -1) # -1 means evaluation, so poison all data in each batch
        poison_size = batch_size if client_id == -1 else int(batch_size * self.poison_ratio)
        
        if poison_size == 0:
            return clean_data, clean_labels
        
        # Use numpy for reproducible poison_indices (uses np.random.seed from main_fl.py)
        # using torch.randperm will cause the poison_indices to be different in each running time
        poison_indices = torch.from_numpy(np.random.permutation(batch_size)[:poison_size])
        # print(f"Poison indices: {poison_indices}")
        poisoned_data = clean_data.clone()
        poisoned_labels = clean_labels.clone()
        
        # Denormalize if needed
        poisoned_data = self._denormalize(poisoned_data) # [0, 1]
        
        # Apply attack-specific poisoning
        self._apply_poison(clean_data, clean_labels, poison_indices, poisoned_data, poisoned_labels)
        # Renormalize if needed
        poisoned_data = self._normalize(poisoned_data) # [-2..., 2...]

        return poisoned_data, poisoned_labels
    
    def _denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data to [0, 1] range"""
        num_channels = data.shape[1]
        if num_channels == 1:
            mean = torch.tensor(self.mean).view(1, 1, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 1, 1, 1).to(data.device)
        else:
            mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 3, 1, 1).to(data.device)
        return data * std + mean
    
    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data using dataset statistics"""
        num_channels = data.shape[1]
        if num_channels == 1:
            mean = torch.tensor(self.mean).view(1, 1, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 1, 1, 1).to(data.device)
        else:
            mean = torch.tensor(self.mean).view(1, 3, 1, 1).to(data.device)
            std = torch.tensor(self.std).view(1, 3, 1, 1).to(data.device)
        return (data - mean) / std
    
    def _apply_poison(self, clean_data: torch.Tensor, clean_labels: torch.Tensor, poison_indices: torch.Tensor, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor) -> None:
        """Apply poison - dispatch based on attack type"""

        if self.use_model_trigger():  # Generative attacks
            # check device of atk_model to match poisoned_data.device
            model_device = next(self.atk_model.parameters()).device
            if model_device != poisoned_data.device:
                self.atk_model = self.atk_model.to(poisoned_data.device)

            self.atk_model.eval()
            with torch.no_grad():
                perturbations, target_labels = self._generate_attack_batch(poisoned_data[poison_indices], poisoned_labels[poison_indices], poisoned_data.device)

            poisoned_data[poison_indices] = perturbations
            poisoned_labels[poison_indices] = target_labels
            
        else:  # Static trigger attacks (e.g., BadNets)
            self._apply_static_trigger(poisoned_data, poisoned_labels, poison_indices)
    
    def use_model_trigger(self) -> bool:
        """Check if attack uses a generative model for trigger generation"""
        return hasattr(self, 'atk_model') and self.atk_model is not None
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor) -> None:
        """Apply static trigger pattern - to be implemented by static trigger attacks"""
        raise NotImplementedError("Static trigger attacks must implement _apply_static_trigger()")
    
    def _generate_attack_batch(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attack-specific perturbation generation - to be implemented by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def get_data_type(self) -> str:
        pass
    
    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor], 
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        """
        Apply model poisoning (e.g., scaling) to model updates.
        
        This method can be overridden by attacks that need to manipulate model updates
        (model poisoning) in addition to data poisoning.
        
        Args:
            local_model_state: Current local model state dict
            global_model_state: Global model state dict
            algorithm: FL algorithm type ('FedAvg', 'FedSGD', 'FedOpt', etc.)
        
        Returns:
            Poisoned model state dict
        """
        # Default implementation: no model poisoning, return local model as-is
        return local_model_state
    
    def should_apply(self, round_idx: int) -> bool:
        if self.attack_frequency == -1:
            # random, if True then apply attack
            return round_idx >= self.attack_start_round and round_idx < self.attack_stop_round
        elif self.attack_frequency == 0:
            # no attack
            return False
        else:
            # attack frequency k, apply attack every k rounds
            return round_idx >= self.attack_start_round and round_idx < self.attack_stop_round and (round_idx - self.attack_start_round) % self.attack_frequency == 0
    
    def _setup_training_atk_model(self, classifier: torch.nn.Module, device: str) -> Tuple[torch.optim.Optimizer, torch.nn.Module, torch.Tensor, torch.Tensor]:
        """Common setup for all attack training"""
        # Move models to device
        self.atk_model.to(device)
        classifier.to(device)
        self.atk_model.train()
        classifier.eval()
        print(f"[{self.__class__.__name__}] with optimizer: {self.atk_optimizer}, lr: {self.atk_lr}")
        # Setup optimizer and loss
        if self.atk_optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.atk_model.parameters(), lr=self.atk_lr)
        elif self.atk_optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.atk_model.parameters(), lr=self.atk_lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.atk_optimizer}")

        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Get normalization parameters
        mean = torch.tensor(self.mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=device).view(1, -1, 1, 1)
        
        return optimizer, loss_fn, mean, std
    
    def _finalize_training(self, verbose: bool, client_id: int, epoch: int, local_ba: float):
        """Common finalization for all attack training"""
        self.atk_model.eval()
        # Move back to CPU to save memory (like global model strategy)
        self.atk_model = self.atk_model.cpu() # TODO: might check this
        
        if verbose:
            print(f"[{self.__class__.__name__}] Client {client_id} Training finished at epoch {epoch}, final backdoor acc={local_ba:.4f}")
    
    def _train_atk_epoch_common(self, trainloader, device: str, mean: torch.Tensor, std: torch.Tensor, 
                           classifier: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module) -> Tuple[float, float]:
        """Common epoch training logic"""
        local_correct, total_loss, total_sample = 0, 0, 0
        
        for idx, batch in enumerate(trainloader):
            # Handle different batch formats
            if isinstance(batch, dict):
                data = batch["image"].to(device)
                labels = batch["label"].to(device)
            else:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
            
            # Unnormalize data
            clean_data = data * std + mean
            
            # Generate attack data (attack-specific)
            atk_data, atk_label = self._generate_attack_batch(clean_data, labels, device)
            
            # Renormalize for classifier
            atk_data = (atk_data - mean) / std
            
            # Forward pass through classifier
            atk_output = classifier(atk_data)
            loss = loss_fn(atk_output, atk_label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = atk_output.argmax(dim=1)
            correct = (pred == atk_label).sum().item()
            local_correct += correct
            total_sample += len(atk_label)
            total_loss += loss.item() * len(atk_label)
            

        avg_loss = total_loss / total_sample if total_sample > 0 else 0.0
        local_ba = local_correct / total_sample if total_sample > 0 else 0.0
        
        return local_ba, avg_loss
        

class BadNetsAttack(BaseAttack):
    """BadNets Attack for Images with proper normalization handling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Auto-set trigger dimensions based on input size (square triggers by default)
        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5  # Default fallback
        
        # Get trigger dimensions (default to square)
        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        
        # Backward compatibility: if trigger_size is provided, use it for both dimensions
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)
        
        self.trigger_pattern = config.get('trigger_pattern', 'square')
        self.target_class = config.get('target_class', 0)
    
    def get_data_type(self) -> str:
        return "image"
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                             poison_indices: torch.Tensor) -> None:
        """Apply BadNets-specific static trigger"""
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        
        # Apply poison to selected indices (bottom-right corner)
        poisoned_data[poison_indices, :, h - self.trigger_height:, w - self.trigger_width:] = 1.0
        poisoned_labels[poison_indices] = self.target_class

class BlendedAttack(BaseAttack):
    """Blended Backdoor Attack: blends a trigger pattern into images"""
    _trigger_cache = None  # Class-level cache for shared trigger
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.blend_alpha = config.get('blend_alpha', 0.3)
        self.target_class = config.get('target_class', 0)

    def get_data_type(self) -> str:
        return "image"
    
    def _get_or_create_trigger(self, device: torch.device, shape: tuple) -> torch.Tensor:
        """Get or create trigger pattern with class-level caching for consistency across clients"""
        if BlendedAttack._trigger_cache is not None and BlendedAttack._trigger_cache.shape == shape:
            return BlendedAttack._trigger_cache.to(device)
        
        # Generate a random trigger pattern (only once for all instances)
        trigger = torch.rand(shape)
        BlendedAttack._trigger_cache = trigger
        return trigger.to(device)
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                             poison_indices: torch.Tensor) -> None:
        """Apply BlendedBackdoor-specific static trigger"""
        # Create or get trigger in pixel space
        trigger = self._get_or_create_trigger(poisoned_data.device, poisoned_data.shape[1:])
        
        # Blend trigger with poisoned data
        for idx in poison_indices:
            poisoned_data[idx] = torch.clamp((1 - self.blend_alpha) * poisoned_data[idx] + self.blend_alpha * trigger, 0.0, 1.0)
            poisoned_labels[idx] = self.target_class



class SinusoidalAttack(BaseAttack):
    """Sinusoidal Backdoor Attack for Images
    
    Adds a sinusoidal intensity pattern across the entire image and flips label to target class.
    The data is assumed denormalized to [0,1] by BaseAttack before this is applied.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_class = config.get('target_class', 0)
        # amplitude of the added sinusoid in [0,1]
        self.sine_amplitude = float(config.get('sine_amplitude', 0.2))
        # number of cycles across the varying dimension (not to confuse with BaseAttack.frequency)
        self.sine_frequency = float(config.get('sine_frequency', 4.0))
        # phase in radians
        self.sine_phase = float(config.get('sine_phase', 0.0))
        # orientation: 'horizontal' (vary along width, constant along height) or 'vertical'
        self.orientation = config.get('sine_orientation', 'horizontal')
        # channel application mode: 'all' to apply to all channels, 'single' to a specific index
        self.channel_mode = config.get('channel_mode', 'all')
        self.channel_index = int(config.get('channel_index', 0))

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                              poison_indices: torch.Tensor) -> None:
        """Apply sinusoidal pattern across the image and set target labels."""
        # poisoned_data: [N, C, H, W] in [0,1]
        _, _, H, W = poisoned_data.shape

        if self.orientation == 'horizontal':
            axis_len = W
            grid = torch.linspace(0, 1, steps=axis_len, device=poisoned_data.device)
            # shape [1, 1, 1, W] broadcast across H
            pattern_1d = torch.sin(2 * torch.pi * self.sine_frequency * grid + self.sine_phase)
            pattern_1d = (pattern_1d + 1.0) / 2.0  # to [0,1]
            pattern = pattern_1d.view(1, 1, 1, W).expand(1, 1, H, W)
        else:  # vertical
            axis_len = H
            grid = torch.linspace(0, 1, steps=axis_len, device=poisoned_data.device)
            # shape [1, 1, H, 1] broadcast across W
            pattern_1d = torch.sin(2 * torch.pi * self.sine_frequency * grid + self.sine_phase)
            pattern_1d = (pattern_1d + 1.0) / 2.0  # to [0,1]
            pattern = pattern_1d.view(1, 1, H, 1).expand(1, 1, H, W)

        # scale by amplitude
        pattern = self.sine_amplitude * pattern

        # apply to selected indices and channels
        if self.channel_mode == 'single':
            # clamp index within available channels
            num_channels = poisoned_data.shape[1]
            ch = max(0, min(self.channel_index, num_channels - 1))
            # create a per-channel mask to add only on selected channel
            add_tensor = torch.zeros_like(poisoned_data[poison_indices])
            add_tensor[:, ch:ch+1, :, :] = pattern
            poisoned_data[poison_indices] = torch.clamp(poisoned_data[poison_indices] + add_tensor, 0.0, 1.0)
        else:
            # apply same pattern to all channels
            poisoned_data[poison_indices] = torch.clamp(poisoned_data[poison_indices] + pattern, 0.0, 1.0)

        poisoned_labels[poison_indices] = self.target_class


class LabelFlippingAttack(BaseAttack):
    """
    Label Flipping Attack
    
    Flips labels without modifying the image data. Supports multiple modes:
    - 'targeted': Flips all samples with source_label to target_label (poison_ratio ignored)
    - 'all2one': Flips poison_ratio of samples to target_label
    - 'all2all': Flips poison_ratio of samples to inverse label (0->9, 1->8, etc.)
    - 'random': Flips poison_ratio of samples to random labels
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Label flipping specific parameters
        self.attack_model = config.get('attack_model', 'targeted')  # targeted, all2one, all2all, random
        self.source_label = config.get('source_label', 2)
        self.target_label = config.get('target_label', 7)
        self.num_classes = config.get('num_classes', 10)  # Number of classes in the dataset
        
        # Validate parameters
        if self.attack_model == 'targeted':
            assert self.source_label != self.target_label, \
                f"Source label ({self.source_label}) and target label ({self.target_label}) must be different"
            assert 0 <= self.source_label < self.num_classes, \
                f"Source label ({self.source_label}) must be in [0, {self.num_classes-1}]"
            assert 0 <= self.target_label < self.num_classes, \
                f"Target label ({self.target_label}) must be in [0, {self.num_classes-1}]"
        elif self.attack_model == 'all2one':
            assert 0 <= self.target_label < self.num_classes, \
                f"Target label ({self.target_label}) must be in [0, {self.num_classes-1}]"
    
    def get_data_type(self) -> str:
        return "image"  # Label flipping works with any data type, but images are most common
    
    def poison_data(self, clean_data: torch.Tensor, clean_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override poison_data to handle targeted mode specially.
        In targeted mode, flip ALL samples with source_label, regardless of poison_ratio.
        """
        batch_size = clean_data.shape[0]
        client_id = self.config.get('client_id', -1)
        
        poisoned_data = clean_data.clone()
        poisoned_labels = clean_labels.clone()
        
        # For targeted mode: flip ALL samples with source_label (ignore poison_ratio)
        if self.attack_model == 'targeted':
            # Find all indices where label matches source_label
            source_indices = (clean_labels == self.source_label).nonzero(as_tuple=True)[0]
            
            if len(source_indices) == 0:
                # No source_label samples in this batch
                return poisoned_data, poisoned_labels
            
            # Denormalize if needed (though we won't modify the data)
            poisoned_data = self._denormalize(poisoned_data)
            
            # Apply label flipping to all source_label samples
            self._apply_static_trigger(poisoned_data, poisoned_labels, source_indices)
            
            # Renormalize
            poisoned_data = self._normalize(poisoned_data)
            
            return poisoned_data, poisoned_labels
        
        # For other modes: use normal poison_ratio logic
        else:
            return super().poison_data(clean_data, clean_labels)
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                             poison_indices: torch.Tensor) -> None:
        """
        Apply label flipping - does not modify image data, only labels.
        """
        if len(poison_indices) == 0:
            return
        
        # Get labels for the poison indices
        labels_to_flip = poisoned_labels[poison_indices]
        
        if self.attack_model == 'targeted':
            # Flip source_label to target_label (already filtered in poison_data for targeted mode)
            poisoned_labels[poison_indices] = self.target_label
            
        elif self.attack_model == 'all2one':
            # Flip all selected labels to target_label
            poisoned_labels[poison_indices] = self.target_label
            
        elif self.attack_model == 'all2all':
            # Flip to inverse label: 0->(num_classes-1), 1->(num_classes-2), etc.
            # Formula: new_label = num_classes - 1 - old_label
            inverse_labels = self.num_classes - 1 - labels_to_flip
            poisoned_labels[poison_indices] = inverse_labels
            
        elif self.attack_model == 'random':
            # Flip to random labels
            random_labels = torch.randint(0, self.num_classes, size=(len(poison_indices),), 
                                         device=poisoned_labels.device, dtype=poisoned_labels.dtype)
            poisoned_labels[poison_indices] = random_labels
            
        else:
            raise ValueError(f"Unknown attack_model: {self.attack_model}. "
                           f"Supported modes: 'targeted', 'all2one', 'all2all', 'random'")
        
        # Note: poisoned_data is not modified - label flipping only changes labels


class ModelReplacementAttack(BaseAttack):
    """
    Model Replacement Attack (Scaling Attack)
    
    [How to Backdoor Federated Learning](https://proceedings.mlr.press/v108/bagdasaryan20a.html) - AISTATS '20
    
    Model replacement attack, also known as constrain-and-scale attack and scaling attack.
    It injects backdoor triggers (data poisoning) and scales model updates (model poisoning).
    
    This implementation includes both:
    1. Data poisoning: Trigger injection (white square pattern at bottom-right corner)
    2. Model poisoning: Model update scaling via apply_model_poisoning() method
    
    The model scaling formula (from FLPoison):
    - For FedAvg: scaled_state = global_state + scaling_factor * (local_state - global_state)
    - For other algorithms: scaled_state = scaling_factor * local_state
    
    Note: In the original paper, the attack also includes:
    - Loss function: alpha * classification_loss + (1-alpha) * anomaly_loss
    - Update scaling: scaled_update = global_weights + scaling_factor * (local_update - global_weights)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model Replacement specific parameters
        self.target_class = config.get('target_class', 6)
        
        # Auto-set trigger dimensions based on input size (square triggers, default 5x5)
        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5  # Default fallback
        
        # Get trigger dimensions (default to square, 5x5 for 32x32 images)
        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        
        # Backward compatibility: if trigger_size is provided, use it for both dimensions
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)
        
        # Note: Model Replacement uses bottom-right corner (same as BadNets in practice)
        # FLPoison's PixelSynthesizer comment says "bottom-left" but the code uses negative indices (-trigger_height, -trigger_width)
        # which actually positions it at bottom-right (same as BadNets)
        self.trigger_position = config.get('trigger_position', 'bottom-right')  # 'bottom-left' or 'bottom-right'
        
        # Model poisoning parameters
        self.scaling_factor = config.get('scaling_factor', 50)  # For model update scaling (default matches FLPoison)
        self.alpha = config.get('alpha', 0.5)  # For loss function: alpha * classification_loss + (1-alpha) * anomaly_loss
    
    def get_data_type(self) -> str:
        return "image"
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                             poison_indices: torch.Tensor) -> None:
        """
        Apply Model Replacement trigger pattern.
        By default, positions trigger at bottom-right corner (same as BadNets).
        """
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        
        # Apply white trigger pattern (value 1.0) to selected indices
        # Default: bottom-right corner (FLPoison's PixelSynthesizer uses negative indices which result in bottom-right)
        if self.trigger_position == 'bottom-right':
            # Bottom-right: from bottom-right corner (default, same as BadNets)
            row_start = h - self.trigger_height
            row_end = h
            col_start = w - self.trigger_width
            col_end = w
        elif self.trigger_position == 'bottom-left':
            # Bottom-left: from bottom-left corner (optional alternative)
            row_start = h - self.trigger_height
            row_end = h
            col_start = 0
            col_end = self.trigger_width
        else:
            raise ValueError(f"Unknown trigger_position: {self.trigger_position}. "
                           f"Supported: 'bottom-left', 'bottom-right'")
        
        # Apply trigger pattern (white square: value 1.0)
        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        # Change labels to target class
        poisoned_labels[poison_indices] = self.target_class
    
    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor], 
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        """
        Apply Model Replacement scaling to model updates.
        
        This implements the model poisoning component of Model Replacement Attack.
        Scales the model update by scaling_factor to replace the global model.
        
        Formula:
        - For FedAvg: scaled_state = global_state + scaling_factor * (local_state - global_state)
        - For other algorithms: scaled_state = scaling_factor * local_state
        
        Args:
            local_model_state: Current local model state dict
            global_model_state: Global model state dict (for FedAvg scaling)
            algorithm: FL algorithm type ('FedAvg', 'FedSGD', 'FedOpt', etc.)
        
        Returns:
            Scaled model state dict
        """
        scaled_state = {}
        
        with torch.no_grad():
            for key in local_model_state.keys():
                local_param = local_model_state[key]
                global_param = global_model_state.get(key, local_param.clone())
                
                if algorithm == 'FedAvg':
                    # For FedAvg: scaled_update = global + scaling_factor * (local - global)
                    # This makes the update replace the global model more effectively
                    update = local_param - global_param
                    scaled_param = global_param + self.scaling_factor * update
                else:
                    # For other algorithms: scaled_update = scaling_factor * local
                    # This scales the entire local model
                    scaled_param = self.scaling_factor * local_param
                
                scaled_state[key] = scaled_param
        
        return scaled_state


class NeurotoxinAttack(BaseAttack):
    """
    Neurotoxin Attack
    
    [Neurotoxin: Durable Backdoors in Federated Learning](https://proceedings.mlr.press/v162/zhang22w.html) - ICML '22
    
    Neurotoxin relies on infrequently updated coordinates by benign clients to hide the backdoor.
    It uses a gradient mask to project gradients to infrequent coordinates, and applies gradient
    norm clipping to prevent excessive updates.
    
    This implementation includes both:
    1. Data poisoning: Trigger injection (white square pattern at bottom-right corner)
    2. Model poisoning: Gradient masking + norm clipping via apply_model_poisoning() method
    
    The model poisoning logic (adapted from FLPoison):
    - Identifies infrequently-updated coordinates (top-k smallest absolute update values)
    - Applies a mask to only update those coordinates
    - Clips the norm of the update to prevent excessive changes
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Neurotoxin specific parameters
        self.target_class = config.get('target_class', 6)
        
        # Auto-set trigger dimensions based on input size (square triggers, default 5x5)
        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5  # Default fallback
        
        # Get trigger dimensions (default to square, 5x5 for 32x32 images)
        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        
        # Backward compatibility: if trigger_size is provided, use it for both dimensions
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)
        
        # Trigger position (default: bottom-right, same as Model Replacement)
        self.trigger_position = config.get('trigger_position', 'bottom-right')
        
        # Model poisoning parameters
        self.topk_ratio = config.get('topk_ratio', 0.1)  # Ratio of top-k smallest absolute values (default matches FLPoison)
        self.norm_threshold = config.get('norm_threshold', 0.2)  # Norm clipping threshold (default matches FLPoison)
    
    def get_data_type(self) -> str:
        return "image"
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                             poison_indices: torch.Tensor) -> None:
        """
        Apply Neurotoxin trigger pattern (same as Model Replacement/BadNets).
        White square pattern positioned at bottom-right corner by default.
        """
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        
        # Apply white trigger pattern (value 1.0) to selected indices
        if self.trigger_position == 'bottom-right':
            # Bottom-right: from bottom-right corner (default)
            row_start = h - self.trigger_height
            row_end = h
            col_start = w - self.trigger_width
            col_end = w
        elif self.trigger_position == 'bottom-left':
            # Bottom-left: from bottom-left corner (optional alternative)
            row_start = h - self.trigger_height
            row_end = h
            col_start = 0
            col_end = self.trigger_width
        else:
            raise ValueError(f"Unknown trigger_position: {self.trigger_position}. "
                           f"Supported: 'bottom-left', 'bottom-right'")
        
        # Apply trigger pattern (white square: value 1.0)
        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        # Change labels to target class
        poisoned_labels[poison_indices] = self.target_class
    
    def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Vectorize model state dict into a flat numpy array"""
        vec_list = []
        for key, param in state_dict.items():
            # Skip non-parameter tensors (e.g., running_mean, running_var, num_batches_tracked)
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])
    
    def _unvectorize_to_state_dict(self, vector: np.ndarray, 
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unvectorize flat numpy array back into model state dict"""
        state_dict = {}
        offset = 0
        
        for key, param in reference_state.items():
            # Skip non-parameter tensors
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
        """
        Apply Neurotoxin gradient masking and norm clipping to model updates.
        
        This implements the model poisoning component of Neurotoxin Attack.
        The method:
        1. Identifies infrequently-updated coordinates (top-k smallest absolute update values)
        2. Applies a mask to only update those coordinates
        3. Clips the norm of the update to prevent excessive changes
        
        Args:
            local_model_state: Current local model state dict
            global_model_state: Global model state dict
            algorithm: FL algorithm type (not used in Neurotoxin, but kept for interface consistency)
        
        Returns:
            Masked and norm-clipped model state dict
        """
        # Compute update: local - global
        update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key:
                continue
            if key in global_model_state:
                update_dict[key] = local_model_state[key] - global_model_state[key]
            else:
                update_dict[key] = local_model_state[key].clone()
        
        # Vectorize the update
        update_vec = self._vectorize_state_dict(update_dict)
        
        if len(update_vec) == 0:
            # If no valid parameters, return local model as-is
            return local_model_state
        
        # Step 1: Create gradient mask (top-k smallest absolute values = infrequent coordinates)
        k = max(1, int(len(update_vec) * self.topk_ratio))
        abs_update_vec = np.abs(update_vec)
        # Get indices of top-k smallest absolute values
        topk_indices = np.argpartition(abs_update_vec, k)[:k]
        
        # Create mask: 1.0 for infrequent coordinates, 0.0 for others
        mask_vec = np.zeros(len(update_vec))
        mask_vec[topk_indices] = 1.0
        
        # Step 2: Apply mask to update
        masked_update_vec = update_vec * mask_vec
        
        # Step 3: Norm clipping
        norm = np.linalg.norm(masked_update_vec)
        if norm > self.norm_threshold:
            scale = self.norm_threshold / norm
            masked_update_vec = masked_update_vec * scale
        
        # Step 4: Unvectorize and add to global state
        masked_update_dict = self._unvectorize_to_state_dict(masked_update_vec, local_model_state)
        
        # Final state: global + masked_clipped_update
        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key:
                    # Keep tracking parameters from local model
                    final_state[key] = local_model_state[key].clone()
                    continue
                
                if key in global_model_state and key in masked_update_dict:
                    final_state[key] = global_model_state[key] + masked_update_dict[key]
                elif key in local_model_state:
                    # Fallback: use local state if global not available
                    final_state[key] = local_model_state[key].clone()
        
        return final_state


class EdgeCaseBackdoorAttack(BaseAttack):
    """
    Edge-case Backdoor Attack
    
    [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084) - NeurIPS '20
    
    Edge-case backdoor attack utilizes edge-case samples from external datasets:
    - ARDIS for MNIST/FashionMNIST (Swedish historical handwritten digits, label 7 → target_label)
    - SouthwestAirline for CIFAR10 (airplane images → target_label)
    
    This implementation includes both:
    1. Data poisoning: Replaces clean samples with edge-case samples from external datasets
    2. Model poisoning: PGD projection + scaling via apply_model_poisoning() method
    
    The model poisoning logic (from FLPoison):
    - PGD projection: Projects update to stay within epsilon ball (L2 or L_inf norm)
    - Scaling attack: Scales the update by scaling_factor (same as Model Replacement)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Edge-case Backdoor specific parameters
        self.target_class = config.get('target_class', 1)
        self.dataset_name = config.get('dataset_name', 'mnist').lower()
        self.data_root = config.get('data_root', './data')
        
        # Model poisoning parameters
        self.epsilon = config.get('epsilon', 0.25)  # PGD epsilon radius (default: 0.25 for MNIST, 0.083 for CIFAR10)
        self.projection_type = config.get('projection_type', 'l_2')  # 'l_2' or 'l_inf'
        self.PGD_attack = config.get('PGD_attack', True)  # Enable PGD projection
        self.scaling_attack = config.get('scaling_attack', True)  # Enable scaling attack
        self.scaling_factor = config.get('scaling_factor', 50)  # Scaling factor (default matches FLPoison)
        self.l2_proj_frequency = config.get('l2_proj_frequency', 1)  # Frequency for L2 projection (default: every epoch)
        
        # Edge-case dataset samples (lazy loading)
        self.edge_case_samples = None
        self.edge_case_labels = None
        self.edge_case_idx = 0  # Index for cycling through edge-case samples
        self._load_edge_case_samples()
    
    def get_data_type(self) -> str:
        return "image"
    
    def _load_edge_case_samples(self):
        """Load edge-case samples from external datasets"""
        try:
            from .edge_case_datasets import load_edge_case_dataset
            self.edge_case_samples, self.edge_case_labels = load_edge_case_dataset(
                self.dataset_name, self.target_class, self.data_root
            )
            # Ensure samples are in the correct format
            if isinstance(self.edge_case_samples, np.ndarray):
                self.edge_case_samples = torch.from_numpy(self.edge_case_samples).float()
            if isinstance(self.edge_case_labels, np.ndarray):
                self.edge_case_labels = torch.from_numpy(self.edge_case_labels).long()
        except (FileNotFoundError, ImportError) as e:
            raise RuntimeError(
                f"Failed to load edge-case dataset for {self.dataset_name}. "
                f"Please ensure edge-case datasets are downloaded and available. "
                f"Error: {e}"
            )
    
    def _apply_poison(self, clean_data: torch.Tensor, clean_labels: torch.Tensor, 
                     poison_indices: torch.Tensor, poisoned_data: torch.Tensor, 
                     poisoned_labels: torch.Tensor) -> None:
        """
        Replace clean samples with edge-case samples.
        This is different from trigger-based attacks - we replace entire samples.
        """
        if self.edge_case_samples is None or len(self.edge_case_samples) == 0:
            raise RuntimeError("Edge-case samples not loaded. Cannot apply poisoning.")
        
        num_to_poison = len(poison_indices)
        
        # Get edge-case samples (cycle through if needed)
        if self.edge_case_idx + num_to_poison <= len(self.edge_case_samples):
            # Enough samples available
            edge_samples = self.edge_case_samples[self.edge_case_idx:self.edge_case_idx + num_to_poison]
            edge_labels = self.edge_case_labels[self.edge_case_idx:self.edge_case_idx + num_to_poison]
            self.edge_case_idx += num_to_poison
        else:
            # Need to cycle through
            remaining = num_to_poison
            edge_samples_list = []
            edge_labels_list = []
            
            while remaining > 0:
                take = min(remaining, len(self.edge_case_samples) - self.edge_case_idx)
                edge_samples_list.append(self.edge_case_samples[self.edge_case_idx:self.edge_case_idx + take])
                edge_labels_list.append(self.edge_case_labels[self.edge_case_idx:self.edge_case_idx + take])
                self.edge_case_idx = (self.edge_case_idx + take) % len(self.edge_case_samples)
                remaining -= take
            
            edge_samples = torch.cat(edge_samples_list, dim=0)
            edge_labels = torch.cat(edge_labels_list, dim=0)
        
        # Ensure edge-case samples have the right shape and device
        # Edge-case samples might be [N, H, W] for MNIST or [N, H, W, C] for CIFAR10
        if edge_samples.dim() == 3:
            # MNIST: [N, H, W] -> [N, 1, H, W]
            edge_samples = edge_samples.unsqueeze(1)
        elif edge_samples.dim() == 4 and edge_samples.shape[1] != 3 and edge_samples.shape[-1] == 3:
            # CIFAR10: [N, H, W, C] -> [N, C, H, W]
            edge_samples = edge_samples.permute(0, 3, 1, 2)
        
        # Normalize edge-case samples to [0, 1] if needed (they should already be)
        edge_samples = torch.clamp(edge_samples, 0.0, 1.0)
        
        # Replace clean samples with edge-case samples
        edge_samples = edge_samples.to(poisoned_data.device)
        edge_labels = edge_labels.to(poisoned_labels.device)
        
        poisoned_data[poison_indices] = edge_samples
        poisoned_labels[poison_indices] = edge_labels
    
    def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Vectorize model state dict into a flat numpy array"""
        vec_list = []
        for key, param in state_dict.items():
            # Skip non-parameter tensors
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])
    
    def _unvectorize_to_state_dict(self, vector: np.ndarray, 
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unvectorize flat numpy array back into model state dict"""
        state_dict = {}
        offset = 0
        
        for key, param in reference_state.items():
            # Skip non-parameter tensors
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
        """
        Apply Edge-case Backdoor PGD projection and scaling to model updates.
        
        This implements the model poisoning component of Edge-case Backdoor Attack.
        The method:
        1. Computes update: local_state - global_state
        2. Applies PGD projection (if enabled): Projects update to epsilon ball (L2 or L_inf)
        3. Applies scaling (if enabled): Scales the update by scaling_factor
        
        Args:
            local_model_state: Current local model state dict
            global_model_state: Global model state dict
            algorithm: FL algorithm type ('FedAvg', 'FedSGD', 'FedOpt', etc.)
        
        Returns:
            PGD-projected and/or scaled model state dict
        """
        # Compute update: local - global
        update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key:
                continue
            if key in global_model_state:
                update_dict[key] = local_model_state[key] - global_model_state[key]
            else:
                update_dict[key] = local_model_state[key].clone()
        
        # Vectorize the update for PGD projection
        update_vec = self._vectorize_state_dict(update_dict)
        
        if len(update_vec) == 0:
            # If no valid parameters, return local model as-is
            return local_model_state
        
        # Get global vector for scaling (needed later)
        global_vec = self._vectorize_state_dict(global_model_state)
        
        # Step 1: Apply PGD projection (if enabled)
        # PGD projects the update (local - global) to stay within epsilon ball
        if self.PGD_attack:
            if self.projection_type == 'l_inf':
                # L_inf projection: clip each coordinate of w_diff to [-epsilon, epsilon]
                # Then reconstruct: projected_local = global + clipped(w_diff)
                smaller_idx = update_vec < -self.epsilon
                larger_idx = update_vec > self.epsilon
                projected_update_vec = update_vec.copy()
                projected_update_vec[smaller_idx] = -self.epsilon
                projected_update_vec[larger_idx] = self.epsilon
            elif self.projection_type == 'l_2':
                # L2 projection: project w_diff to epsilon ball if norm > epsilon
                w_diff_norm = np.linalg.norm(update_vec)
                if w_diff_norm > self.epsilon:
                    projected_update_vec = self.epsilon * update_vec / w_diff_norm
                else:
                    projected_update_vec = update_vec
            else:
                raise ValueError(f"Unknown projection_type: {self.projection_type}. "
                               f"Supported: 'l_2', 'l_inf'")
        else:
            # No PGD projection, use update as-is
            projected_update_vec = update_vec
        
        # Step 2: Apply scaling (if enabled)
        # Note: In FLPoison, scaling is applied after PGD projection
        # projected_update_vec is (local - global) after PGD projection
        if self.scaling_attack:
            if algorithm == 'FedAvg':
                # For FedAvg: scaled_update = global + scaling_factor * (local - global)
                # Since projected_update_vec is (local - global) after PGD, we scale it
                scaled_update_vec = self.scaling_factor * projected_update_vec
            else:
                # For other algorithms: scaled_update = scaling_factor * update
                # Where update is the local model (after PGD), not (local - global)
                # So: projected_local = global + projected_update
                # scaled_update = scaling_factor * projected_local
                # But we need to return as (scaled_local - global)
                projected_local_vec = global_vec + projected_update_vec
                scaled_local_vec = self.scaling_factor * projected_local_vec
                scaled_update_vec = scaled_local_vec - global_vec
        else:
            # No scaling, use projected update as-is
            scaled_update_vec = projected_update_vec
        
        # Step 3: Unvectorize and add to global state
        scaled_update_dict = self._unvectorize_to_state_dict(scaled_update_vec, local_model_state)
        
        # Final state: global + scaled_update
        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key:
                    # Keep tracking parameters from local model
                    final_state[key] = local_model_state[key].clone()
                    continue
                
                if key in global_model_state and key in scaled_update_dict:
                    final_state[key] = global_model_state[key] + scaled_update_dict[key]
                elif key in local_model_state:
                    # Fallback: use local state if global not available
                    final_state[key] = local_model_state[key].clone()
        
        return final_state


class ThreeDFedAttack(BaseAttack):
    """
    3DFed Attack
    
    [3DFed: Adaptive and Extensible Framework for Covert Backdoor Attack in Federated Learning](https://ieeexplore.ieee.org/document/10179401) - S&P '23
    
    3DFed is a covert backdoor attack framework that combines data poisoning (trigger injection)
    with model poisoning (norm clipping) to evade detection. The attack:
    1. Uses BadNets-style trigger injection for data poisoning
    2. Clips backdoor update norm to match benign update norm (model poisoning)
    3. Adaptively tunes attack parameters based on feedback
    
    This implementation includes:
    1. Data poisoning: BadNets-style trigger injection (white square pattern)
    2. Model poisoning: Norm clipping via apply_model_poisoning() method
    
    Note: The original implementation in FLPoison is incomplete. This implementation
    includes the core functionality: trigger injection + norm clipping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 3DFed specific parameters
        self.target_class = config.get('target_class', 0)
        
        # Auto-set trigger dimensions based on input size (square triggers, default 5x5)
        input_dim = config.get('input_dim', 32)
        if input_dim == 28:
            default_size = 4
        elif input_dim == 32:
            default_size = 5
        elif input_dim == 64:
            default_size = 9
        else:
            default_size = 5  # Default fallback
        
        # Get trigger dimensions (default to square, 5x5 for 32x32 images)
        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)
        
        # Backward compatibility: if trigger_size is provided, use it for both dimensions
        if 'trigger_size' in config:
            self.trigger_height = config.get('trigger_size', default_size)
            self.trigger_width = config.get('trigger_size', default_size)
        
        # Trigger position (default: bottom-right, same as BadNets)
        self.trigger_position = config.get('trigger_position', 'bottom-right')
        
        # Model poisoning parameters
        self.scaling_factor = config.get('scaling_factor', 1.0)  # Scaling factor for norm clipping
        self.use_norm_clipping = config.get('use_norm_clipping', True)  # Enable norm clipping
    
    def get_data_type(self) -> str:
        return "image"
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, 
                             poison_indices: torch.Tensor) -> None:
        """
        Apply 3DFed trigger pattern (same as BadNets).
        White square pattern positioned at bottom-right corner by default.
        """
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
        
        # Apply white trigger pattern (value 1.0) to selected indices
        if self.trigger_position == 'bottom-right':
            # Bottom-right: from bottom-right corner (default)
            row_start = h - self.trigger_height
            row_end = h
            col_start = w - self.trigger_width
            col_end = w
        elif self.trigger_position == 'bottom-left':
            # Bottom-left: from bottom-left corner (optional alternative)
            row_start = h - self.trigger_height
            row_end = h
            col_start = 0
            col_end = self.trigger_width
        else:
            raise ValueError(f"Unknown trigger_position: {self.trigger_position}. "
                           f"Supported: 'bottom-left', 'bottom-right'")
        
        # Apply trigger pattern (white square: value 1.0)
        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        # Change labels to target class
        poisoned_labels[poison_indices] = self.target_class
    
    def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """Vectorize model state dict into a flat numpy array"""
        vec_list = []
        for key, param in state_dict.items():
            # Skip non-parameter tensors
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])
    
    def _unvectorize_to_state_dict(self, vector: np.ndarray, 
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unvectorize flat numpy array back into model state dict"""
        state_dict = {}
        offset = 0
        
        for key, param in reference_state.items():
            # Skip non-parameter tensors
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
        """
        Apply 3DFed norm clipping to model updates.
        
        This implements the model poisoning component of 3DFed Attack.
        The method clips the backdoor update norm to match the benign update norm,
        making the attack more covert.
        
        Args:
            local_model_state: Current local model state dict (backdoor model)
            global_model_state: Global model state dict
            algorithm: FL algorithm type ('FedAvg', 'FedSGD', 'FedOpt', etc.)
        
        Returns:
            Norm-clipped model state dict
        """
        if not self.use_norm_clipping:
            # No norm clipping, return local model as-is
            return local_model_state
        
        # Compute backdoor update: local - global
        backdoor_update_dict = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key:
                continue
            if key in global_model_state:
                backdoor_update_dict[key] = local_model_state[key] - global_model_state[key]
            else:
                backdoor_update_dict[key] = local_model_state[key].clone()
        
        # Vectorize updates for norm computation
        backdoor_update_vec = self._vectorize_state_dict(backdoor_update_dict)
        
        if len(backdoor_update_vec) == 0:
            # If no valid parameters, return local model as-is
            return local_model_state
        
        # Compute backdoor update norm
        backdoor_norm = np.linalg.norm(backdoor_update_vec)
        
        # For norm clipping, we need a reference benign norm
        # In the original 3DFed (FLPoison), it uses the norm of benign updates from normal training
        # The original logic (from FLPoison threedfed.py lines 70-81):
        #   1. Clip if backdoor_norm > benign_norm: backdoor_update = backdoor_update * (benign_norm / backdoor_norm)
        #   2. scale_factor = min((benign_norm / backdoor_norm), self.scaling_factor)
        #   3. return max(scale_factor, 1) * backdoor_update
        #
        # Since we don't have direct access to benign updates in this framework,
        # we use a reference_norm from config (which should be the benign update norm)
        reference_norm = self.config.get('reference_norm', None)
        
        if reference_norm is not None:
            # Apply norm clipping following FLPoison's logic (threedfed.py lines 70-81)
            benign_norm = reference_norm
            
            # Step 1: Clip if backdoor_norm > benign_norm
            # Original: if backdoor_norm > benign_norm: backdoor_update = backdoor_update * (benign_norm / backdoor_norm)
            if backdoor_norm > benign_norm:
                backdoor_update_vec = backdoor_update_vec * (benign_norm / backdoor_norm)
                # Update backdoor_norm after clipping
                backdoor_norm = benign_norm  # After clipping, backdoor_norm = benign_norm
            
            # Step 2: Compute scale_factor = min((benign_norm / backdoor_norm), scaling_factor)
            # Original: scale_factor = min((benign_norm / backdoor_norm), self.args.scaling_factor)
            # After clipping (if applied), benign_norm / backdoor_norm = 1
            # Otherwise, benign_norm / backdoor_norm >= 1 (since we didn't clip)
            scale_factor = min((benign_norm / backdoor_norm) if backdoor_norm > 0 else 1.0, 
                             self.scaling_factor)
            
            # Step 3: Apply final scale: max(scale_factor, 1) * backdoor_update
            # Original: return max(scale_factor, 1) * backdoor_update
            final_scale = max(scale_factor, 1.0)
            backdoor_update_vec = backdoor_update_vec * final_scale
        
        # Unvectorize and add to global state
        clipped_update_dict = self._unvectorize_to_state_dict(backdoor_update_vec, local_model_state)
        
        # Final state: global + clipped_update
        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                if 'num_batches_tracked' in key:
                    # Keep tracking parameters from local model
                    final_state[key] = local_model_state[key].clone()
                    continue
                
                if key in global_model_state and key in clipped_update_dict:
                    final_state[key] = global_model_state[key] + clipped_update_dict[key]
                elif key in local_model_state:
                    # Fallback: use local state if global not available
                    final_state[key] = local_model_state[key].clone()
        
        return final_state


class DBAAttack(BaseAttack):
    """
    Distributed Backdoor Attack (DBA)
    
    Each client implants a local trigger during training, and the global trigger 
    is composed of all local triggers during inference.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # DBA specific parameters
        self.target_class = config.get('target_class', 0)
        self.trigger_nums = config.get('trigger_nums', 4)  # Number of local triggers
        self.use_global_backdoor = config.get('use_global_backdoor', False)  # Global vs distributed backdoor
        self.trigger_height = config.get('trigger_height', 1)
        self.trigger_width = config.get('trigger_width', 6)
        self.trigger_gap = config.get('trigger_gap', 2)
        self.trigger_shift = config.get('trigger_shift', 0)

        # # Auto-set trigger dimensions based on input size (horizontal triggers)
        # input_dim = config.get('input_dim', 32)
        # if input_dim == 28:
        #     self.trigger_height = config.get('trigger_height', 1)   # Horizontal: height < width
        #     self.trigger_width = config.get('trigger_width', 4)
        #     self.trigger_gap = config.get('trigger_gap', 2)
        # elif input_dim == 32:
        #     self.trigger_height = config.get('trigger_height', 1)   # Horizontal: height < width
        #     self.trigger_width = config.get('trigger_width', 6)
        #     self.trigger_gap = config.get('trigger_gap', 3)
        # elif input_dim == 64:
        #     self.trigger_height = config.get('trigger_height', 2)   # Horizontal: height < width
        #     self.trigger_width = config.get('trigger_width', 10)
        #     self.trigger_gap = config.get('trigger_gap', 2)
        # else:
        #     # Default fallback
        #     self.trigger_height = config.get('trigger_height', 1)
        #     self.trigger_width = config.get('trigger_width', 6)
        #     self.trigger_gap = config.get('trigger_gap', 2)
        
        
        # get apply_to_client_ids
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])
        # Create local triggers
        self._create_local_triggers()
    
    def _create_local_triggers(self):
        """Create 4 local triggers arranged in 2x2 pattern"""
        # Pattern (horizontal triggers):
        # **** ****
        # **** ****
        # **** ****
        # **** ****
        # https://github.com/AI-secure/DBA/blob/master/utils/mnist_params.yaml
        # input dim 28x28: trigger size 4*1, trigger gap 2, trigger shift 0
        # input dim 32x32: trigger size 6*1, trigger gap 3, trigger shift 0
        # input dim 64x64: trigger size 10*2, trigger gap 2, trigger shift 0
        
        num_channels = len(self.mean)
        
        # Create triggers in [0,1] range (no normalization needed since data is denormalized)
        self.local_triggers = []
        for i in range(self.trigger_nums):
            # Each trigger is a rectangular pattern (width x height)
            if num_channels == 1:
                trigger = torch.ones((1, 1, self.trigger_height, self.trigger_width))
            else:
                trigger = torch.ones((1, num_channels, self.trigger_height, self.trigger_width))
            self.local_triggers.append(trigger)
        
        # Pre-compute all trigger positions for efficient vectorized application
        self._precompute_trigger_positions()
    
    def _precompute_trigger_positions(self):
        """Pre-compute trigger positions for efficient vectorized application"""
        self.trigger_positions = []
        for i in range(self.trigger_nums):
            row_start, col_start = self._setup_trigger_position(i)
            row_end = row_start + self.trigger_height
            col_end = col_start + self.trigger_width
            self.trigger_positions.append((row_start, row_end, col_start, col_end))
    
    def _setup_trigger_position(self, trigger_idx: int) -> Tuple[int, int]:
        """Setup trigger position for given trigger index (2x2 grid layout)"""
        # Calculate position based on 2x2 grid
        # trigger_idx: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        
        # Row position: 0 for top row (idx 0,1), trigger_height+gap for bottom row (idx 2,3)
        row_starter = (trigger_idx // 2) * (self.trigger_height + self.trigger_gap) + self.trigger_shift
        
        # Column position: 0 for left column (idx 0,2), trigger_width+gap for right column (idx 1,3)  
        column_starter = (trigger_idx % 2) * (self.trigger_width + self.trigger_gap) + self.trigger_shift
        
        return row_starter, column_starter
    
    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor) -> None:
        """Apply DBA trigger pattern"""
        client_id = self.config.get('client_id', -1)
        
        # Check if we should use global backdoor or distributed backdoor
        if client_id == -1 or self.use_global_backdoor:
            # Inference phase or global backdoor mode: apply all triggers (global trigger)
            self._apply_global_trigger(poisoned_data, poisoned_labels, poison_indices)
            # self._apply_local_trigger(poisoned_data, poisoned_labels, poison_indices, self.apply_to_client_ids[3])
        else:
            # Training phase with distributed backdoor: apply only local trigger based on client_id
            self._apply_local_trigger(poisoned_data, poisoned_labels, poison_indices, client_id)
    
    def _apply_local_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor, client_id: int) -> None:
        """Apply local trigger for training phase - vectorized version"""
        # get the index of client_id in apply_to_client_ids
        trigger_idx = self.apply_to_client_ids.index(client_id) % self.trigger_nums
        # print(f"🔍 DBA Debug - Client {client_id}, Trigger {trigger_idx}:")
        trigger = self.local_triggers[trigger_idx].to(poisoned_data.device)
        
        # Get pre-computed trigger position
        row_start, row_end, col_start, col_end = self.trigger_positions[trigger_idx]
        
        # Debug: print trigger dimensions and position
        # if client_id == 0:  # Only print for first client to avoid spam
        #     print(f"🔍 DBA Debug - Client {client_id}, Trigger {trigger_idx}:")
        #     print(f"   Trigger shape: {trigger.shape}")
        #     print(f"   Position: row[{row_start}:{row_end}], col[{col_start}:{col_end}]")
        #     print(f"   Trigger dimensions: height={self.trigger_height}, width={self.trigger_width}")
        
        # Ensure trigger fits within image bounds
        row_end = min(row_end, poisoned_data.shape[2])
        col_end = min(col_end, poisoned_data.shape[3])
        
        if row_end > row_start and col_end > col_start:
            # Vectorized application - apply trigger to all poisoned samples at once
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = trigger[0, :, :actual_height, :actual_width]
            # Change labels to target class
            poisoned_labels[poison_indices] = self.target_class
    
    def _apply_global_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor, poison_indices: torch.Tensor) -> None:
        """Apply global trigger (all local triggers) for inference phase - vectorized version"""
        for trigger_idx in range(self.trigger_nums):
            trigger = self.local_triggers[trigger_idx].to(poisoned_data.device)
            
            # Get pre-computed trigger position
            row_start, row_end, col_start, col_end = self.trigger_positions[trigger_idx]
            
            # Ensure trigger fits within image bounds
            row_end = min(row_end, poisoned_data.shape[2])
            col_end = min(col_end, poisoned_data.shape[3])
            
            if row_end > row_start and col_end > col_start:
                # Vectorized application - apply trigger to all poisoned samples at once
                actual_height = row_end - row_start
                actual_width = col_end - col_start
                poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = trigger[0, :, :actual_height, :actual_width]
        
        # Change labels to target class (only once after applying all triggers)
        poisoned_labels[poison_indices] = self.target_class
    
    def get_data_type(self) -> str:
        return "image"
    

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
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
