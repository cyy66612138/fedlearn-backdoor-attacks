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
    
    # def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
    #                           global_model_state: Dict[str, torch.Tensor],
    #                           algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
    #     """
    #     Apply Model Replacement scaling to model updates.
    #
    #     This implements the model poisoning component of Model Replacement Attack.
    #     Scales the model update by scaling_factor to replace the global model.
    #
    #     Formula:
    #     - For FedAvg: scaled_state = global_state + scaling_factor * (local_state - global_state)
    #     - For other algorithms: scaled_state = scaling_factor * local_state
    #
    #     Args:
    #         local_model_state: Current local model state dict
    #         global_model_state: Global model state dict (for FedAvg scaling)
    #         algorithm: FL algorithm type ('FedAvg', 'FedSGD', 'FedOpt', etc.)
    #
    #     Returns:
    #         Scaled model state dict
    #     """
    #     scaled_state = {}
    #
    #     with torch.no_grad():
    #         for key in local_model_state.keys():
    #             local_param = local_model_state[key]
    #             global_param = global_model_state.get(key, local_param.clone())
    #
    #             if algorithm == 'FedAvg':
    #                 # For FedAvg: scaled_update = global + scaling_factor * (local - global)
    #                 # This makes the update replace the global model more effectively
    #                 update = local_param - global_param
    #                 scaled_param = global_param + self.scaling_factor * update
    #             else:
    #                 # For other algorithms: scaled_update = scaling_factor * local
    #                 # This scales the entire local model
    #                 scaled_param = self.scaling_factor * local_param
    #
    #             scaled_state[key] = scaled_param
    #
    #     return scaled_state
    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        """
        Apply Model Replacement scaling to model updates. (Bug-Fixed Version)
        修复了导致非 FedAvg 算法下全局权重被放大 50 倍的灾难性 Bug，
        并增加了对 BN 层非浮点参数的保护。
        """
        scaled_state = {}

        with torch.no_grad():
            for key in local_model_state.keys():
                # =================================================================
                # 🛠️ BUG FIX 1: 绝对不能缩放 Batch Normalization 层的统计参数！
                # 否则会导致模型内部前向传播时的均值和方差彻底错乱，引发 NaN。
                # =================================================================
                if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                    scaled_state[key] = local_model_state[key].clone()
                    continue

                local_param = local_model_state[key]
                global_param = global_model_state.get(key, local_param.clone())

                # =================================================================
                # 🛠️ BUG FIX 2: 严格遵守差值缩放逻辑 (Strict Delta Scaling)
                # 无论服务器用的是 FedAvg 还是 DnC、RLR，服务器提取梯度的方法都是 local - global。
                # 所以我们必须始终通过放大更新量 (Delta) 来伪造 local_state。
                # 绝对不能直接 `50 * local_param`！
                # =================================================================

                # 计算更新量 Delta
                update = local_param.float() - global_param.float()

                # 伪造放大后的客户端模型权重
                scaled_param = global_param.float() + self.scaling_factor * update

                # 还原至原始的数据类型
                scaled_state[key] = scaled_param.to(local_param.dtype)

        return scaled_state

# class NeurotoxinAttack(BaseAttack):
#     """
#     Neurotoxin Attack
#
#     [Neurotoxin: Durable Backdoors in Federated Learning](https://proceedings.mlr.press/v162/zhang22w.html) - ICML '22
#
#     Neurotoxin relies on infrequently updated coordinates by benign clients to hide the backdoor.
#     It uses a gradient mask to project gradients to infrequent coordinates, and applies gradient
#     norm clipping to prevent excessive updates.
#
#     This implementation includes both:
#     1. Data poisoning: Trigger injection (white square pattern at bottom-right corner)
#     2. Model poisoning: Gradient masking + norm clipping via apply_model_poisoning() method
#
#     The model poisoning logic (adapted from FLPoison):
#     - Identifies infrequently-updated coordinates (top-k smallest absolute update values)
#     - Applies a mask to only update those coordinates
#     - Clips the norm of the update to prevent excessive changes
#     """
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#
#         # Neurotoxin specific parameters
#         self.target_class = config.get('target_class', 6)
#
#         # Auto-set trigger dimensions based on input size (square triggers, default 5x5)
#         input_dim = config.get('input_dim', 32)
#         if input_dim == 28:
#             default_size = 4
#         elif input_dim == 32:
#             default_size = 5
#         elif input_dim == 64:
#             default_size = 9
#         else:
#             default_size = 5  # Default fallback
#
#         # Get trigger dimensions (default to square, 5x5 for 32x32 images)
#         self.trigger_height = config.get('trigger_height', default_size)
#         self.trigger_width = config.get('trigger_width', default_size)
#
#         # Backward compatibility: if trigger_size is provided, use it for both dimensions
#         if 'trigger_size' in config:
#             self.trigger_height = config.get('trigger_size', default_size)
#             self.trigger_width = config.get('trigger_size', default_size)
#
#         # Trigger position (default: bottom-right, same as Model Replacement)
#         self.trigger_position = config.get('trigger_position', 'bottom-right')
#
#         # Model poisoning parameters
#         self.topk_ratio = config.get('topk_ratio', 0.1)  # Ratio of top-k smallest absolute values (default matches FLPoison)
#         self.norm_threshold = config.get('norm_threshold', 0.2)  # Norm clipping threshold (default matches FLPoison)
#
#     def get_data_type(self) -> str:
#         return "image"
#
#     def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
#                              poison_indices: torch.Tensor) -> None:
#         """
#         Apply Neurotoxin trigger pattern (same as Model Replacement/BadNets).
#         White square pattern positioned at bottom-right corner by default.
#         """
#         _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]
#
#         # Apply white trigger pattern (value 1.0) to selected indices
#         if self.trigger_position == 'bottom-right':
#             # Bottom-right: from bottom-right corner (default)
#             row_start = h - self.trigger_height
#             row_end = h
#             col_start = w - self.trigger_width
#             col_end = w
#         elif self.trigger_position == 'bottom-left':
#             # Bottom-left: from bottom-left corner (optional alternative)
#             row_start = h - self.trigger_height
#             row_end = h
#             col_start = 0
#             col_end = self.trigger_width
#         else:
#             raise ValueError(f"Unknown trigger_position: {self.trigger_position}. "
#                            f"Supported: 'bottom-left', 'bottom-right'")
#
#         # Apply trigger pattern (white square: value 1.0)
#         poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
#         # Change labels to target class
#         poisoned_labels[poison_indices] = self.target_class
#
#     def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
#         """Vectorize model state dict into a flat numpy array"""
#         vec_list = []
#         for key, param in state_dict.items():
#             # Skip non-parameter tensors (e.g., running_mean, running_var, num_batches_tracked)
#             if 'num_batches_tracked' in key:
#                 continue
#             vec_list.append(param.detach().cpu().numpy().flatten())
#         return np.concatenate(vec_list) if vec_list else np.array([])
#
#     def _unvectorize_to_state_dict(self, vector: np.ndarray,
#                                    reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Unvectorize flat numpy array back into model state dict"""
#         state_dict = {}
#         offset = 0
#
#         for key, param in reference_state.items():
#             # Skip non-parameter tensors
#             if 'num_batches_tracked' in key:
#                 state_dict[key] = param.clone()
#                 continue
#
#             numel = param.numel()
#             param_vec = vector[offset:offset + numel]
#             state_dict[key] = torch.from_numpy(param_vec.reshape(param.shape)).to(param.device, dtype=param.dtype)
#             offset += numel
#
#         return state_dict
#
#     def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
#                               global_model_state: Dict[str, torch.Tensor],
#                               algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
#         """
#         Apply Neurotoxin gradient masking and norm clipping to model updates.
#
#         This implements the model poisoning component of Neurotoxin Attack.
#         The method:
#         1. Identifies infrequently-updated coordinates (top-k smallest absolute update values)
#         2. Applies a mask to only update those coordinates
#         3. Clips the norm of the update to prevent excessive changes
#
#         Args:
#             local_model_state: Current local model state dict
#             global_model_state: Global model state dict
#             algorithm: FL algorithm type (not used in Neurotoxin, but kept for interface consistency)
#
#         Returns:
#             Masked and norm-clipped model state dict
#         """
#         # Compute update: local - global
#         update_dict = {}
#         for key in local_model_state.keys():
#             if 'num_batches_tracked' in key:
#                 continue
#             if key in global_model_state:
#                 update_dict[key] = local_model_state[key] - global_model_state[key]
#             else:
#                 update_dict[key] = local_model_state[key].clone()
#
#         # Vectorize the update
#         update_vec = self._vectorize_state_dict(update_dict)
#
#         if len(update_vec) == 0:
#             # If no valid parameters, return local model as-is
#             return local_model_state
#
#         # Step 1: Create gradient mask (top-k smallest absolute values = infrequent coordinates)
#         k = max(1, int(len(update_vec) * self.topk_ratio))
#         abs_update_vec = np.abs(update_vec)
#         # Get indices of top-k smallest absolute values
#         topk_indices = np.argpartition(abs_update_vec, k)[:k]
#
#         # Create mask: 1.0 for infrequent coordinates, 0.0 for others
#         mask_vec = np.zeros(len(update_vec))
#         mask_vec[topk_indices] = 1.0
#
#         # Step 2: Apply mask to update
#         masked_update_vec = update_vec * mask_vec
#
#         # Step 3: Norm clipping
#         norm = np.linalg.norm(masked_update_vec)
#         if norm > self.norm_threshold:
#             scale = self.norm_threshold / norm
#             masked_update_vec = masked_update_vec * scale
#
#         # Step 4: Unvectorize and add to global state
#         masked_update_dict = self._unvectorize_to_state_dict(masked_update_vec, local_model_state)
#
#         # Final state: global + masked_clipped_update
#         final_state = {}
#         with torch.no_grad():
#             for key in local_model_state.keys():
#                 if 'num_batches_tracked' in key:
#                     # Keep tracking parameters from local model
#                     final_state[key] = local_model_state[key].clone()
#                     continue
#
#                 if key in global_model_state and key in masked_update_dict:
#                     final_state[key] = global_model_state[key] + masked_update_dict[key]
#                 elif key in local_model_state:
#                     # Fallback: use local state if global not available
#                     final_state[key] = local_model_state[key].clone()
#
#         return final_state
class NeurotoxinAttack(BaseAttack):
    """
    Neurotoxin Attack (自适应动态裁剪升级版)

    [Neurotoxin: Durable Backdoors in Federated Learning] - ICML '22

    改进说明：
    废弃了原版代码中死板的 norm_threshold = 0.2 绝对阈值。
    改为在本地计算每次更新的真实良性范数 (benign_norm) 作为动态安全阈值。
    在执行掩码后，允许对残余的后门梯度进行适度放大，但最终范数会被严格裁剪至不大于 benign_norm，
    从而在实现“最高隐蔽性”的同时，尽可能保证后门的致死率。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Neurotoxin 基础参数
        self.target_class = config.get('target_class', 6)

        # 触发器尺寸与位置设置 (默认 5x5 右下角)
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

        # 模型投毒参数
        self.topk_ratio = config.get('topk_ratio', 0.1)  # 休眠参数比例 (寻找绝对值最小的 10%)
        # 新增：由于掩码会丢失大量能量，允许先放大，再通过自适应裁剪保证安全
        self.lambda_val = config.get('lambda_val', 5.0)

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                              poison_indices: torch.Tensor) -> None:
        """
        数据投毒：植入静态触发器
        """
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
        """将模型字典展平为一维 Numpy 数组"""
        vec_list = []
        for key, param in state_dict.items():
            if 'num_batches_tracked' in key:
                continue
            vec_list.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(vec_list) if vec_list else np.array([])

    def _unvectorize_to_state_dict(self, vector: np.ndarray,
                                   reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将一维数组还原回模型字典结构"""
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

    # def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
    #                           global_model_state: Dict[str, torch.Tensor],
    #                           algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
    #     """
    #     模型投毒核心逻辑：动态寻找休眠参数 -> 掩码与放大 -> 自适应范数裁剪
    #     """
    #     update_dict = {}
    #     for key in local_model_state.keys():
    #         if 'num_batches_tracked' in key:
    #             continue
    #         if key in global_model_state:
    #             update_dict[key] = local_model_state[key] - global_model_state[key]
    #         else:
    #             update_dict[key] = local_model_state[key].clone()
    #
    #     # 1. 展平本地更新量
    #     update_vec = self._vectorize_state_dict(update_dict)
    #     if len(update_vec) == 0:
    #         return local_model_state
    #
    #     # ==========================================
    #     # 🟢 改进 1：计算当前轮次的“自适应安全阈值”
    #     # ==========================================
    #     benign_norm = np.linalg.norm(update_vec)
    #
    #     # 2. 寻找最不活跃的休眠坐标 (Top-k smallest absolute values)
    #     k = max(1, int(len(update_vec) * self.topk_ratio))
    #     abs_update_vec = np.abs(update_vec)
    #     topk_indices = np.argpartition(abs_update_vec, k)[:k]
    #
    #     # 3. 生成掩码
    #     mask_vec = np.zeros(len(update_vec))
    #     mask_vec[topk_indices] = 1.0
    #
    #     # ==========================================
    #     # 🟢 改进 2：掩码后施加能量补偿，并执行自适应裁剪
    #     # ==========================================
    #     # 因为清零了 90% 的参数，为了保证后门有效性，先尝试放大 lambda_val 倍
    #     masked_update_vec = update_vec * mask_vec * self.lambda_val
    #
    #     # 计算当前恶意更新的范数
    #     current_norm = np.linalg.norm(masked_update_vec)
    #
    #     # 【核心安全锁】：如果放大后的能量超过了原有的良性范数，强行裁剪回 benign_norm
    #     # 保证服务器无论设置多么严格的范数异常检测，都绝对抓不住它
    #     if current_norm > benign_norm and current_norm > 0:
    #         scale = benign_norm / current_norm
    #         masked_update_vec = masked_update_vec * scale
    #
    #     # 4. 还原结构并伪装成完整模型
    #     masked_update_dict = self._unvectorize_to_state_dict(masked_update_vec, local_model_state)
    #
    #     final_state = {}
    #     with torch.no_grad():
    #         for key in local_model_state.keys():
    #             if 'num_batches_tracked' in key:
    #                 final_state[key] = local_model_state[key].clone()
    #                 continue
    #
    #             if key in global_model_state and key in masked_update_dict:
    #                 final_state[key] = global_model_state[key] + masked_update_dict[key]
    #             elif key in local_model_state:
    #                 final_state[key] = local_model_state[key].clone()
    #
    #     return final_state
    def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
        """
        模型投毒核心逻辑：动态寻找休眠参数 -> 掩码与放大 -> 自适应范数裁剪 + 数值安全保障
        """
        update_dict = {}

        # 1. 计算本地更新量 (Delta)
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key:
                continue
            # 确保在 CPU 上进行 numpy 转换前的计算，避免显存爆炸
            if key in global_model_state:
                update_dict[key] = (local_model_state[key].cpu() - global_model_state[key].cpu()).float()
            else:
                update_dict[key] = local_model_state[key].cpu().clone().float()

        # 2. 展平本地更新量
        update_vec = self._vectorize_state_dict(update_dict)
        if len(update_vec) == 0:
            return local_model_state

        # ==========================================
        # 🟢 改进 1：数值清理 (防止初始更新中就包含 NaN)
        # ==========================================
        update_vec = np.nan_to_num(update_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算当前轮次的“自适应安全阈值” (良性范数)
        benign_norm = np.linalg.norm(update_vec)
        if benign_norm == 0:
            return local_model_state

        # 3. 寻找最不活跃的休眠坐标 (Top-k smallest absolute values)
        # Neurotoxin 核心：寻找那些在正常训练中几乎不更新的参数
        k = max(1, int(len(update_vec) * self.topk_ratio))
        abs_update_vec = np.abs(update_vec)
        # 获取绝对值最小的前 k 个索引
        topk_indices = np.argpartition(abs_update_vec, k)[:k]

        # 4. 生成掩码并执行能量补偿
        # 仅保留这 10% 的休眠参数，并尝试放大 lambda_val 倍
        masked_update_vec = np.zeros_like(update_vec)
        masked_update_vec[topk_indices] = update_vec[topk_indices] * self.lambda_val

        # ==========================================
        # 🟢 改进 2：自适应范数裁剪 (核心安全锁)
        # ==========================================
        current_norm = np.linalg.norm(masked_update_vec)

        # 如果放大后的恶意更新能量超过了原始更新的范数，执行投影裁剪
        # 这能确保更新量在统计特征上与良性客户端一致，且不会数值爆炸
        if current_norm > benign_norm and current_norm > 0:
            scale = benign_norm / current_norm
            masked_update_vec = masked_update_vec * scale

        # ==========================================
        # 🟢 改进 3：二次数值安全检查 (防止产生新的 NaN)
        # ==========================================
        masked_update_vec = np.nan_to_num(masked_update_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. 还原结构并伪装成完整模型
        masked_update_dict = self._unvectorize_to_state_dict(masked_update_vec, local_model_state)

        final_state = {}
        with torch.no_grad():
            for key in local_model_state.keys():
                # 保持 BN 层统计信息不变
                if 'num_batches_tracked' in key:
                    final_state[key] = local_model_state[key].clone()
                    continue

                if key in global_model_state and key in masked_update_dict:
                    # 最终参数 = 全局基座 + 篡改后的休眠层增量
                    # 确保转回原始数据类型 (如 float16/float32)
                    final_state[key] = global_model_state[key].cpu() + masked_update_dict[key].to(
                        global_model_state[key].dtype)
                elif key in local_model_state:
                    final_state[key] = local_model_state[key].cpu().clone()

        return final_state

# class EdgeCaseBackdoorAttack(BaseAttack):
#     """
#     Edge-case Backdoor Attack
#
#     [Attack of the Tails: Yes, You Really Can Backdoor Federated Learning](https://arxiv.org/abs/2007.05084) - NeurIPS '20
#
#     Edge-case backdoor attack utilizes edge-case samples from external datasets:
#     - ARDIS for MNIST/FashionMNIST (Swedish historical handwritten digits, label 7 → target_label)
#     - SouthwestAirline for CIFAR10 (airplane images → target_label)
#
#     This implementation includes both:
#     1. Data poisoning: Replaces clean samples with edge-case samples from external datasets
#     2. Model poisoning: PGD projection + scaling via apply_model_poisoning() method
#
#     The model poisoning logic (from FLPoison):
#     - PGD projection: Projects update to stay within epsilon ball (L2 or L_inf norm)
#     - Scaling attack: Scales the update by scaling_factor (same as Model Replacement)
#     """
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#
#         # Edge-case Backdoor specific parameters
#         self.target_class = config.get('target_class', 1)
#         self.dataset_name = config.get('dataset_name', 'mnist').lower()
#         self.data_root = config.get('data_root', './data')
#
#         # Model poisoning parameters
#         self.epsilon = config.get('epsilon', 0.25)  # PGD epsilon radius (default: 0.25 for MNIST, 0.083 for CIFAR10)
#         self.projection_type = config.get('projection_type', 'l_2')  # 'l_2' or 'l_inf'
#         self.PGD_attack = config.get('PGD_attack', True)  # Enable PGD projection
#         self.scaling_attack = config.get('scaling_attack', True)  # Enable scaling attack
#         self.scaling_factor = config.get('scaling_factor', 50)  # Scaling factor (default matches FLPoison)
#         self.l2_proj_frequency = config.get('l2_proj_frequency', 1)  # Frequency for L2 projection (default: every epoch)
#
#         # Edge-case dataset samples (lazy loading)
#         self.edge_case_samples = None
#         self.edge_case_labels = None
#         self.edge_case_idx = 0  # Index for cycling through edge-case samples
#         self._load_edge_case_samples()
#
#     def get_data_type(self) -> str:
#         return "image"
#
#     def _load_edge_case_samples(self):
#         """Load edge-case samples from external datasets"""
#         try:
#             from .edge_case_datasets import load_edge_case_dataset
#             self.edge_case_samples, self.edge_case_labels = load_edge_case_dataset(
#                 self.dataset_name, self.target_class, self.data_root
#             )
#             # Ensure samples are in the correct format
#             if isinstance(self.edge_case_samples, np.ndarray):
#                 self.edge_case_samples = torch.from_numpy(self.edge_case_samples).float()
#             if isinstance(self.edge_case_labels, np.ndarray):
#                 self.edge_case_labels = torch.from_numpy(self.edge_case_labels).long()
#         except (FileNotFoundError, ImportError) as e:
#             raise RuntimeError(
#                 f"Failed to load edge-case dataset for {self.dataset_name}. "
#                 f"Please ensure edge-case datasets are downloaded and available. "
#                 f"Error: {e}"
#             )
#
#     def _apply_poison(self, clean_data: torch.Tensor, clean_labels: torch.Tensor,
#                      poison_indices: torch.Tensor, poisoned_data: torch.Tensor,
#                      poisoned_labels: torch.Tensor) -> None:
#         """
#         Replace clean samples with edge-case samples.
#         This is different from trigger-based attacks - we replace entire samples.
#         """
#         if self.edge_case_samples is None or len(self.edge_case_samples) == 0:
#             raise RuntimeError("Edge-case samples not loaded. Cannot apply poisoning.")
#
#         num_to_poison = len(poison_indices)
#
#         # Get edge-case samples (cycle through if needed)
#         if self.edge_case_idx + num_to_poison <= len(self.edge_case_samples):
#             # Enough samples available
#             edge_samples = self.edge_case_samples[self.edge_case_idx:self.edge_case_idx + num_to_poison]
#             edge_labels = self.edge_case_labels[self.edge_case_idx:self.edge_case_idx + num_to_poison]
#             self.edge_case_idx += num_to_poison
#         else:
#             # Need to cycle through
#             remaining = num_to_poison
#             edge_samples_list = []
#             edge_labels_list = []
#
#             while remaining > 0:
#                 take = min(remaining, len(self.edge_case_samples) - self.edge_case_idx)
#                 edge_samples_list.append(self.edge_case_samples[self.edge_case_idx:self.edge_case_idx + take])
#                 edge_labels_list.append(self.edge_case_labels[self.edge_case_idx:self.edge_case_idx + take])
#                 self.edge_case_idx = (self.edge_case_idx + take) % len(self.edge_case_samples)
#                 remaining -= take
#
#             edge_samples = torch.cat(edge_samples_list, dim=0)
#             edge_labels = torch.cat(edge_labels_list, dim=0)
#
#         # Ensure edge-case samples have the right shape and device
#         # Edge-case samples might be [N, H, W] for MNIST or [N, H, W, C] for CIFAR10
#         if edge_samples.dim() == 3:
#             # MNIST: [N, H, W] -> [N, 1, H, W]
#             edge_samples = edge_samples.unsqueeze(1)
#         elif edge_samples.dim() == 4 and edge_samples.shape[1] != 3 and edge_samples.shape[-1] == 3:
#             # CIFAR10: [N, H, W, C] -> [N, C, H, W]
#             edge_samples = edge_samples.permute(0, 3, 1, 2)
#
#         # Normalize edge-case samples to [0, 1] if needed (they should already be)
#         edge_samples = torch.clamp(edge_samples, 0.0, 1.0)
#
#         # Replace clean samples with edge-case samples
#         edge_samples = edge_samples.to(poisoned_data.device)
#         edge_labels = edge_labels.to(poisoned_labels.device)
#
#         poisoned_data[poison_indices] = edge_samples
#         poisoned_labels[poison_indices] = edge_labels
#
#     def _vectorize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
#         """Vectorize model state dict into a flat numpy array"""
#         vec_list = []
#         for key, param in state_dict.items():
#             # Skip non-parameter tensors
#             if 'num_batches_tracked' in key:
#                 continue
#             vec_list.append(param.detach().cpu().numpy().flatten())
#         return np.concatenate(vec_list) if vec_list else np.array([])
#
#     def _unvectorize_to_state_dict(self, vector: np.ndarray,
#                                    reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Unvectorize flat numpy array back into model state dict"""
#         state_dict = {}
#         offset = 0
#
#         for key, param in reference_state.items():
#             # Skip non-parameter tensors
#             if 'num_batches_tracked' in key:
#                 state_dict[key] = param.clone()
#                 continue
#
#             numel = param.numel()
#             param_vec = vector[offset:offset + numel]
#             state_dict[key] = torch.from_numpy(param_vec.reshape(param.shape)).to(param.device, dtype=param.dtype)
#             offset += numel
#
#         return state_dict
#
#     def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
#                               global_model_state: Dict[str, torch.Tensor],
#                               algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
#         """
#         Apply Edge-case Backdoor PGD projection and scaling to model updates.
#
#         This implements the model poisoning component of Edge-case Backdoor Attack.
#         The method:
#         1. Computes update: local_state - global_state
#         2. Applies PGD projection (if enabled): Projects update to epsilon ball (L2 or L_inf)
#         3. Applies scaling (if enabled): Scales the update by scaling_factor
#
#         Args:
#             local_model_state: Current local model state dict
#             global_model_state: Global model state dict
#             algorithm: FL algorithm type ('FedAvg', 'FedSGD', 'FedOpt', etc.)
#
#         Returns:
#             PGD-projected and/or scaled model state dict
#         """
#         # Compute update: local - global
#         update_dict = {}
#         for key in local_model_state.keys():
#             if 'num_batches_tracked' in key:
#                 continue
#             if key in global_model_state:
#                 update_dict[key] = local_model_state[key] - global_model_state[key]
#             else:
#                 update_dict[key] = local_model_state[key].clone()
#
#         # Vectorize the update for PGD projection
#         update_vec = self._vectorize_state_dict(update_dict)
#
#         if len(update_vec) == 0:
#             # If no valid parameters, return local model as-is
#             return local_model_state
#
#         # Get global vector for scaling (needed later)
#         global_vec = self._vectorize_state_dict(global_model_state)
#
#         # Step 1: Apply PGD projection (if enabled)
#         # PGD projects the update (local - global) to stay within epsilon ball
#         if self.PGD_attack:
#             if self.projection_type == 'l_inf':
#                 # L_inf projection: clip each coordinate of w_diff to [-epsilon, epsilon]
#                 # Then reconstruct: projected_local = global + clipped(w_diff)
#                 smaller_idx = update_vec < -self.epsilon
#                 larger_idx = update_vec > self.epsilon
#                 projected_update_vec = update_vec.copy()
#                 projected_update_vec[smaller_idx] = -self.epsilon
#                 projected_update_vec[larger_idx] = self.epsilon
#             elif self.projection_type == 'l_2':
#                 # L2 projection: project w_diff to epsilon ball if norm > epsilon
#                 w_diff_norm = np.linalg.norm(update_vec)
#                 if w_diff_norm > self.epsilon:
#                     projected_update_vec = self.epsilon * update_vec / w_diff_norm
#                 else:
#                     projected_update_vec = update_vec
#             else:
#                 raise ValueError(f"Unknown projection_type: {self.projection_type}. "
#                                f"Supported: 'l_2', 'l_inf'")
#         else:
#             # No PGD projection, use update as-is
#             projected_update_vec = update_vec
#
#         # Step 2: Apply scaling (if enabled)
#         # Note: In FLPoison, scaling is applied after PGD projection
#         # projected_update_vec is (local - global) after PGD projection
#         if self.scaling_attack:
#             if algorithm == 'FedAvg':
#                 # For FedAvg: scaled_update = global + scaling_factor * (local - global)
#                 # Since projected_update_vec is (local - global) after PGD, we scale it
#                 scaled_update_vec = self.scaling_factor * projected_update_vec
#             else:
#                 # For other algorithms: scaled_update = scaling_factor * update
#                 # Where update is the local model (after PGD), not (local - global)
#                 # So: projected_local = global + projected_update
#                 # scaled_update = scaling_factor * projected_local
#                 # But we need to return as (scaled_local - global)
#                 projected_local_vec = global_vec + projected_update_vec
#                 scaled_local_vec = self.scaling_factor * projected_local_vec
#                 scaled_update_vec = scaled_local_vec - global_vec
#         else:
#             # No scaling, use projected update as-is
#             scaled_update_vec = projected_update_vec
#
#         # Step 3: Unvectorize and add to global state
#         scaled_update_dict = self._unvectorize_to_state_dict(scaled_update_vec, local_model_state)
#
#         # Final state: global + scaled_update
#         final_state = {}
#         with torch.no_grad():
#             for key in local_model_state.keys():
#                 if 'num_batches_tracked' in key:
#                     # Keep tracking parameters from local model
#                     final_state[key] = local_model_state[key].clone()
#                     continue
#
#                 if key in global_model_state and key in scaled_update_dict:
#                     final_state[key] = global_model_state[key] + scaled_update_dict[key]
#                 elif key in local_model_state:
#                     # Fallback: use local state if global not available
#                     final_state[key] = local_model_state[key].clone()
#
#         return final_state
class EdgeCaseBackdoorAttack(BaseAttack):
    """
    Edge-case Backdoor Attack (Bug-Fixed Version)
    修复了导致模型崩溃的暴力缩放 Bug 与导致特征断层的归一化 Bug。
    """

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

        # [安全修复]：必须使用 .clone()，防止原地修改污染缓存的源数据集
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

        # =================================================================
        # 🛠️ BUG FIX 1: 归一化对齐 (Normalization Alignment)
        # 必须使用与主线数据相同的 Mean 和 Std 对 Edge-case 样本进行归一化，否则模型无法识别特征
        # =================================================================
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

        # Step 1: PGD Projection
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

        # =================================================================
        # 🛠️ BUG FIX 2: 严格更新量缩放 (Strict Delta Scaling)
        # 无论什么算法，绝对不能去缩放 global_weights 本身！只准缩放更新量 (delta)！
        # =================================================================
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


class FedDAREAttack(BaseAttack):
    """
    FedDARE Attack: Drop-And-REscale for extreme sparse model poisoning.
    Data poisoning uses a standard static trigger (like BadNets).
    Model poisoning drops p% of the update and rescales the rest by 1/(1-p).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.target_class = config.get('target_class', 0)
        self.drop_rate = config.get('drop_rate', 0.99)  # 默认丢弃 99% 的梯度参数

        # 触发器设置 (复用 BadNets 的右下角白块)
        input_dim = config.get('input_dim', 32)
        default_size = 5 if input_dim >= 32 else 4
        self.trigger_height = config.get('trigger_height', default_size)
        self.trigger_width = config.get('trigger_width', default_size)

    def get_data_type(self) -> str:
        return "image"

    def _apply_static_trigger(self, poisoned_data: torch.Tensor, poisoned_labels: torch.Tensor,
                              poison_indices: torch.Tensor) -> None:
        """植入后门触发器 (数据投毒阶段)"""
        _, h, w = poisoned_data.shape[1], poisoned_data.shape[2], poisoned_data.shape[3]

        row_start = h - self.trigger_height
        row_end = h
        col_start = w - self.trigger_width
        col_end = w

        # 植入白色触发器并将标签翻转为 target_class
        poisoned_data[poison_indices, :, row_start:row_end, col_start:col_end] = 1.0
        poisoned_labels[poison_indices] = self.target_class

    # def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
    #                           global_model_state: Dict[str, torch.Tensor],
    #                           algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
    #     """
    #     FedDARE 核心逻辑: 掩码稀疏化 + 能量重缩放 (模型投毒阶段)
    #     """
    #     dare_state = {}
    #     scale_factor = 1.0 / (1.0 - self.drop_rate)
    #
    #     with torch.no_grad():
    #         for key in local_model_state.keys():
    #             # 跳过 BatchNorm 的统计层参数
    #             if 'num_batches_tracked' in key:
    #                 dare_state[key] = local_model_state[key].clone()
    #                 continue
    #
    #             local_param = local_model_state[key]
    #             global_param = global_model_state.get(key, local_param.clone())
    #
    #             # 1. 计算原始恶意更新量 (Delta W)
    #             update = local_param - global_param
    #
    #             # 2. 生成同维度的独立伯努利掩码 (0 或 1)
    #             # torch.rand_like 生成 [0,1) 的均匀分布，大于 drop_rate 的部分置为 1 (存活)
    #             mask = (torch.rand_like(update) > self.drop_rate).float()
    #
    #             # 3. DARE 核心操作：丢弃 + 缩放放大
    #             dare_update = update * mask * scale_factor
    #
    #             # 4. 加回全局模型，构造最终上传的恶意模型权重
    #             dare_state[key] = global_param + dare_update
    #
    #     return dare_state
    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm="FedAvg"):
        """
        正确的 FedDARE 模型投毒：只放大 Delta (更新量)，且保证绝对安全
        """
        poisoned_state = {}

        # 计算缩放系数 (例如 drop_rate=0.9, 则放大 10 倍)
        scale_factor = 1.0 / (1.0 - self.drop_rate + 1e-8)

        for key in local_model_state.keys():
            # 跳过 Batch Normalization 的追踪统计量（放大它们必死无疑）
            if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                poisoned_state[key] = local_model_state[key]
                continue

            # 1. 计算真实的增量 Delta
            delta = local_model_state[key].float() - global_model_state[key].float()

            # 2. 生成与 Delta 形状相同的随机掩码 (Drop)
            # 大于 drop_rate 的设为 1 (保留)，小于的设为 0 (丢弃)
            mask = (torch.rand_like(delta) > self.drop_rate).float()

            # 3. 对 Delta 进行掩码并重缩放 (Rescale)
            poisoned_delta = delta * mask * scale_factor

            # [安全锁] 防止单层梯度过大导致 NaN (截断极大值)
            # 这也是对抗服务器范数裁剪 (Norm Clipping) 的关键一步
            # max_delta_norm = 5.0  # 设定一个安全的阈值，你可以根据情况调整
            # current_norm = torch.norm(poisoned_delta)
            # if current_norm > max_delta_norm:
            #     poisoned_delta = poisoned_delta * (max_delta_norm / (current_norm + 1e-8))

            # 4. 将投毒后的 Delta 加回全局模型，得到最终要上传的欺骗性参数
            poisoned_state[key] = (global_model_state[key].float() + poisoned_delta).to(local_model_state[key].dtype)

        return poisoned_state


# class LayerwisePoisoningAttack(BadNetsAttack):
#     """
#     Implementation of the Layer-wise Poisoning (LP) attack from the paper:
#     'BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS'
#
#     This attack identifies Backdoor-Critical (BC) layers and only uploads
#     poisoned updates for those specific layers. For non-BC layers, it uploads
#     the clean global model parameters (zero-delta) to maintain high stealthiness
#     and bypass distance-based or similarity-based anomaly detection.
#     """
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         # Default BC layers for ResNet18 (e.g., the last block and fully connected layer)
#         self.bc_layers = config.get('bc_layers', ['layer4', 'linear', 'fc'])
#         self.lambda_val = config.get('lambda_val', 2.0)  # Scaling factor for BC layers
#
#     def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
#                               global_model_state: Dict[str, torch.Tensor],
#                               algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
#
#         poisoned_state = {}
#
#         # Calculate normal bounds to safely scale if necessary
#         raw_squared_sum = 0.0
#         for key in local_model_state.keys():
#             if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
#                 continue
#             delta = local_model_state[key].float() - global_model_state[key].float()
#             raw_squared_sum += torch.sum(delta ** 2).item()
#         estimated_benign_norm = (raw_squared_sum ** 0.5)
#         max_allowed_norm = estimated_benign_norm * 1.5
#
#         # Poison only the BC layers
#         poisoned_deltas = {}
#         poisoned_squared_sum = 0.0
#
#         for key in local_model_state.keys():
#             if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
#                 continue
#
#             delta = local_model_state[key].float() - global_model_state[key].float()
#
#             # Check if this layer is a Backdoor-Critical (BC) layer
#             is_bc_layer = any(bc_name in key for bc_name in self.bc_layers)
#
#             if is_bc_layer:
#                 # [核心逻辑]: 仅对 BC 层投毒并放大 (lambda_val)
#                 p_delta = delta * self.lambda_val
#             else:
#                 # [核心逻辑]: 非 BC 层伪装为正常 (返回0更新)
#                 # 论文中提到非关键层使用正常模型参数平均值填充。由于客户端无法获取其他人的更新，
#                 # 返回原始全局模型参数(即0更新)是最完美的隐蔽策略。
#                 p_delta = torch.zeros_like(delta)
#
#             poisoned_deltas[key] = p_delta
#             poisoned_squared_sum += torch.sum(p_delta ** 2).item()
#
#         poisoned_norm = (poisoned_squared_sum ** 0.5)
#
#         # 自适应裁剪 (Adaptive scaling to evade detection)
#         clip_rate = 1.0
#         if poisoned_norm > max_allowed_norm and poisoned_norm > 0:
#             clip_rate = max_allowed_norm / poisoned_norm
#
#         # Reconstruct the model
#         for key in local_model_state.keys():
#             if key not in poisoned_deltas:
#                 poisoned_state[key] = local_model_state[key]
#             else:
#                 final_delta = poisoned_deltas[key] * clip_rate
#                 poisoned_state[key] = (global_model_state[key].float() + final_delta).to(local_model_state[key].dtype)
#
#         return poisoned_state

# class LayerwisePoisoningAttack(BadNetsAttack):
#     """
#     Implementation of the Layer-wise Poisoning (LP) attack from the paper:
#     'BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS'
#
#     动态寻找 Backdoor-Critical (BC) 层：
#     通过评估本地后门训练后各层的更新量范数（Norm），自动挑选更新最剧烈的前 K 层作为 BC 层，
#     仅对这些层进行放大投毒，其余层返回 0 更新以维持隐蔽性。
#     """
#
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         # 废弃固定的 bc_layers，改为按比例或指定数量动态选取
#         # 默认选取更新范数最大的前 10% 的参数矩阵作为 BC 层
#         self.bc_layer_ratio = config.get('bc_layer_ratio', 0.1)
#         self.num_bc_layers = config.get('num_bc_layers', None)  # 也可以在 YAML 中直接指定具体层数 (例如 6)
#         self.lambda_val = config.get('lambda_val', 2.0)
#
#     def apply_model_poisoning(self, local_model_state: Dict[str, torch.Tensor],
#                               global_model_state: Dict[str, torch.Tensor],
#                               algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:
#
#         poisoned_state = {}
#         layer_updates = {}
#         layer_norms = {}
#         raw_squared_sum = 0.0
#
#         # 1. 计算所有层的本地更新量 (Delta W) 及其 L2 范数
#         for key in local_model_state.keys():
#             # 跳过 BN 层的统计特征
#             if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
#                 continue
#
#             delta = local_model_state[key].float() - global_model_state[key].float()
#             layer_updates[key] = delta
#
#             # 使用范数来衡量该层对后门任务的“敏感度/关键度”
#             norm = torch.norm(delta, p=2).item()
#             layer_norms[key] = norm
#             raw_squared_sum += norm ** 2
#
#         estimated_benign_norm = (raw_squared_sum ** 0.5)
#         max_allowed_norm = estimated_benign_norm * 1.5
#
#         # 2. 动态定位 Backdoor-Critical (BC) 层
#         # 按照各层更新量的大小进行降序排序
#         sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
#
#         # 决定要选择的 BC 层数量 (K)
#         if self.num_bc_layers is not None:
#             k = self.num_bc_layers
#         else:
#             k = max(1, int(len(layer_norms) * self.bc_layer_ratio))
#
#         # 提取 Top-K 最敏感的层名作为 BC 层
#         bc_layers_keys = set([layer[0] for layer in sorted_layers[:k]])
#
#         # 3. 针对性执行层级投毒 (Layer-wise Poisoning)
#         poisoned_deltas = {}
#         poisoned_squared_sum = 0.0
#
#         for key in local_model_state.keys():
#             if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
#                 continue
#
#             delta = layer_updates[key]
#
#             if key in bc_layers_keys:
#                 # [核心逻辑]: 关键层 -> 放大后门更新，植入恶意特征
#                 p_delta = delta * self.lambda_val
#             else:
#                 # [核心逻辑]: 非关键层 -> 强行掩码清零，骗过相似度检测 (如 FoolsGold)
#                 p_delta = torch.zeros_like(delta)
#
#             poisoned_deltas[key] = p_delta
#             poisoned_squared_sum += torch.sum(p_delta ** 2).item()
#
#         poisoned_norm = (poisoned_squared_sum ** 0.5)
#
#         # 4. 自适应裁剪 (防止范数超标被服务器截断)
#         clip_rate = 1.0
#         if poisoned_norm > max_allowed_norm and poisoned_norm > 0:
#             clip_rate = max_allowed_norm / poisoned_norm
#
#         # 5. 重构并封装最终要上传的欺骗性模型参数
#         for key in local_model_state.keys():
#             if key not in poisoned_deltas:
#                 poisoned_state[key] = local_model_state[key]
#             else:
#                 final_delta = poisoned_deltas[key] * clip_rate
#                 poisoned_state[key] = (global_model_state[key].float() + final_delta).to(local_model_state[key].dtype)
#
#         return poisoned_state

import torch
from typing import Dict, Any, List


class LayerwisePoisoningAttack(BadNetsAttack):
    """
    Layer-wise Poisoning (LP) Attack - [严格动态每轮 LSA 搜索版]
    论文: BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS (ICLR 2024)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.lambda_val = config.get('lambda_val', 2.0)
        self.lsa_bsr_threshold = config.get('lsa_bsr_threshold', 0.5)

        # 用于接收 Client 临时注入的环境变量
        self.lsa_model = None
        self.lsa_dataloader = None
        self.lsa_device = None

    def setup_lsa_environment(self, model: torch.nn.Module, dataloader: Any, device: torch.device):
        """
        供 Client 在外部调用的依赖注入接口。
        把模型实例、本地数据集和设备传进来，供 LSA 做物理前向传播测试使用。
        """
        self.lsa_model = model
        self.lsa_dataloader = dataloader
        self.lsa_device = device

    # def _run_dynamic_lsa(self, benign_state: Dict[str, torch.Tensor],
    #                      malicious_state: Dict[str, torch.Tensor]) -> List[str]:
    #     """
    #     核心 LSA 算法：每一轮动态执行，真实测试 BSR
    #     """
    #     if self.lsa_model is None or self.lsa_dataloader is None:
    #         raise ValueError(
    #             "LSA Environment not setup! Please call setup_lsa_environment() in client.py before applying model poisoning.")
    #
    #     bc_layers_found = []
    #     self.lsa_model.eval()
    #
    #     # 内部评估函数：在干净 batch 上动态投毒并测试 BSR
    #     def evaluate_bsr(eval_model):
    #         correct, total = 0, 0
    #         with torch.no_grad():
    #             # 为了节省每轮巨大的计算开销，LSA 通常只取 1-2 个 Batch 进行快速嗅探
    #             for batch_idx, (data, target) in enumerate(self.lsa_dataloader):
    #                 if batch_idx > 1: break  # 取前 2 个 Batch 足矣
    #
    #                 data, target = data.to(self.lsa_device), target.to(self.lsa_device)
    #                 # 动态生成后门数据 (复用 BadNets 的画白块逻辑)
    #                 atk_data, atk_label = self.poison_data(data, target)
    #
    #                 outputs = eval_model(atk_data)
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 total += atk_label.size(0)
    #                 correct += (predicted == atk_label).sum().item()
    #         return correct / total if total > 0 else 0.0
    #
    #     print(f"[{self.name}] Running In-situ LSA for dynamic BC layers discovery...")
    #
    #     for layer_name in benign_state.keys():
    #         if 'num_batches_tracked' in layer_name or 'running_mean' in layer_name or 'running_var' in layer_name:
    #             continue
    #
    #         # 1. 构造 Hybrid 模型 (全局干净底座 + 单层恶意参数)
    #         hybrid_state = {k: v.clone() for k, v in benign_state.items()}
    #         hybrid_state[layer_name] = malicious_state[layer_name].clone()
    #
    #         # 2. 物理测试 BSR
    #         self.lsa_model.load_state_dict(hybrid_state)
    #         bsr = evaluate_bsr(self.lsa_model)
    #
    #         # 3. 筛选 BC 层
    #         if bsr > self.lsa_bsr_threshold:
    #             bc_layers_found.append(layer_name)
    #
    #     # 恢复模型原本的权重，防止弄脏内存
    #     self.lsa_model.load_state_dict(malicious_state)
    #     print(f"[{self.name}] LSA completed. Found {len(bc_layers_found)} BC layers this round: {bc_layers_found}")
    #
    #     # 防御性编程：如果这轮没找到任何 BC 层，默认兜底使用最深的全连接层
    #     if not bc_layers_found:
    #         fallback = [k for k in benign_state.keys() if 'fc' in k or 'linear' in k or 'classifier' in k]
    #         return fallback
    #
    #     return bc_layers_found

    def _run_dynamic_lsa(self, benign_state: Dict[str, torch.Tensor],
                         malicious_state: Dict[str, torch.Tensor]) -> List[str]:
        """
        核心 LSA 算法：在 GPU 上极速动态执行，真实测试纯粹的 BSR
        """
        if self.lsa_model is None or self.lsa_dataloader is None:
            raise ValueError("LSA Environment not setup!")

        # 🚀 加速关键 1：把模型强制搬回 GPU
        self.lsa_model = self.lsa_model.to(self.lsa_device)
        self.lsa_model.eval()

        def evaluate_pure_bsr(eval_model):
            correct, total = 0, 0
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(self.lsa_dataloader):
                    # 🚀 加速关键 2：只测 1 个 Batch (约128张图) 就绝对足够暴露后门了
                    if batch_idx > 0: break

                    data = data.to(self.lsa_device)

                    # 强制 100% 贴图
                    poisoned_data = data.clone()
                    poison_indices = torch.arange(data.shape[0])

                    poisoned_data = self._denormalize(poisoned_data)
                    dummy_labels = torch.zeros(data.shape[0], dtype=torch.long)
                    self._apply_static_trigger(poisoned_data, dummy_labels, poison_indices)
                    poisoned_data = self._normalize(poisoned_data)

                    # GPU 极速预测
                    outputs = eval_model(poisoned_data)
                    _, predicted = torch.max(outputs.data, 1)

                    total += data.size(0)
                    correct += (predicted == self.target_class).sum().item()

            return correct / total if total > 0 else 0.0

        print(f"[{self.name}] Running In-situ LSA on GPU for dynamic BC layers discovery...")

        layer_bsr_scores = {}

        for layer_name in benign_state.keys():
            if 'num_batches_tracked' in layer_name or 'running_mean' in layer_name or 'running_var' in layer_name:
                continue

            # 构造 Hybrid 模型并加载 (PyTorch 会自动把 CPU dict 传到 GPU 模型里)
            hybrid_state = {k: v.clone() for k, v in benign_state.items()}
            hybrid_state[layer_name] = malicious_state[layer_name].clone()

            self.lsa_model.load_state_dict(hybrid_state)
            layer_bsr_scores[layer_name] = evaluate_pure_bsr(self.lsa_model)

        # 🚀 内存安全：测完之后把模型物归原主，并放回 CPU，防止主进程显存爆炸
        self.lsa_model.load_state_dict(malicious_state)
        self.lsa_model = self.lsa_model.cpu()

        # 降序排列，取 Top-K 最致命的层
        sorted_layers = sorted(layer_bsr_scores.items(), key=lambda x: x[1], reverse=True)

        bc_ratio = self.config.get('bc_layer_ratio', 0.05)
        k = max(2, int(len(sorted_layers) * bc_ratio))

        bc_layers_found = [layer_name for layer_name, score in sorted_layers[:k]]

        print(f"[{self.name}] LSA completed in < 1s. Top {k} BC layers this round: {bc_layers_found}")

        return bc_layers_found
    def apply_model_poisoning(self,
                              local_model_state: Dict[str, torch.Tensor],
                              global_model_state: Dict[str, torch.Tensor],
                              algorithm: str = 'FedAvg') -> Dict[str, torch.Tensor]:

        # ==========================================
        # 💥 杀招 0：在篡改前，当场执行 LSA 算出本轮名单
        # ==========================================
        dynamic_bc_layers = self._run_dynamic_lsa(global_model_state, local_model_state)

        poisoned_state = {}
        for key in local_model_state.keys():
            if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                poisoned_state[key] = local_model_state[key].clone()
                continue

            # 使用刚才实时算出来的 dynamic_bc_layers 进行判断
            is_bc_layer = key in dynamic_bc_layers

            if is_bc_layer:
                # 💥 杀招 1：关键层放大
                delta = local_model_state[key].float() - global_model_state[key].float()
                poisoned_delta = delta * self.lambda_val
                poisoned_state[key] = (global_model_state[key].float() + poisoned_delta).to(
                    local_model_state[key].dtype)
            else:
                # 💥 杀招 2：非关键层隐蔽 (Delta=0)
                poisoned_state[key] = global_model_state[key].clone()

        return poisoned_state


import torch


class MinMaxAttack:
    def __init__(self, dev_type='std'):
        """
        dev_type: 扰动向量的类型，可选 'std' (标准差反向), 'sign' (符号反向), 'unit' (单位向量反向)
        论文中最常用且对 Non-IID 效果最好的是 'std'
        """
        self.dev_type = dev_type

    def _flatten_weights(self, weights_dict):
        """将 state_dict 字典展平为 1D Tensor"""
        return torch.cat([v.flatten() for v in weights_dict.values()])

    def _unflatten_weights(self, flat_tensor, template_dict):
        """将 1D Tensor 还原为 state_dict 字典"""
        unflat_dict = {}
        idx = 0
        for k, v in template_dict.items():
            numel = v.numel()
            unflat_dict[k] = flat_tensor[idx:idx + numel].view_as(v).clone()
            idx += numel
        return unflat_dict

    def apply_attack(self, malicious_local_updates):
        """
        malicious_local_updates: List[dict]
        包含所有恶意客户端在本地正常训练后的权重更新（用作估算全局良性分布的代理）
        """
        # 1. 展平所有恶意客户端的权重
        flat_updates = [self._flatten_weights(w) for w in malicious_local_updates]
        stacked_updates = torch.stack(flat_updates)  # Shape: (num_attackers, num_params)

        # 2. 估算良性分布的均值 (mu)
        mu = torch.mean(stacked_updates, dim=0)

        # 3. 计算扰动方向 (v_p)
        if self.dev_type == 'sign':
            deviation = -torch.sign(mu)
        elif self.dev_type == 'std':
            deviation = -torch.std(stacked_updates, dim=0)
        elif self.dev_type == 'unit':
            deviation = -mu / (torch.norm(mu) + 1e-8)
        else:
            deviation = -torch.std(stacked_updates, dim=0)

        # 4. 计算“安全半径”（任意两个本地更新之间的最大欧式距离）
        distances = torch.cdist(stacked_updates, stacked_updates, p=2.0)
        max_distance = torch.max(distances)

        # 5. 二分查找 (Binary Search) 寻找最大的 gamma
        gamma_succ = 0.0
        gamma_fail = 100.0  # 初始上限，可根据具体模型收敛情况调大
        gamma = gamma_fail / 2.0
        threshold = 1e-4

        while abs(gamma_fail - gamma_succ) > threshold:
            # 构造候选恶意向量
            candidate_malicious = mu + gamma * deviation

            # 计算候选向量到所有已知本地更新的距离
            dist_to_locals = torch.norm(stacked_updates - candidate_malicious, dim=1)

            # 如果候选向量到所有已知更新的距离都在安全半径内
            if torch.max(dist_to_locals) <= max_distance:
                gamma_succ = gamma
                gamma = gamma + (gamma_fail - gamma) / 2.0
            else:
                gamma_fail = gamma
                gamma = gamma - (gamma - gamma_succ) / 2.0

        # 6. 生成最终的恶意更新向量
        best_malicious_flat = mu + gamma_succ * deviation

        # 7. 还原为 state_dict 并返回（所有恶意客户端将上传这同一个字典）
        template = malicious_local_updates[0]
        malicious_state_dict = self._unflatten_weights(best_malicious_flat, template)

        return malicious_state_dict


import torch


class TrimAttack:
    """
    专门针对 Trimmed Mean 和 Median 聚合规则的局部模型投毒攻击
    """

    def __init__(self, num_attackers=1, num_total_clients=10, b=2.0):
        self.num_attackers = num_attackers
        self.num_total = num_total_clients
        self.b = b  # 控制参数在边缘的系数，默认 2 倍标准差

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
        # 1. 展平并计算参考分布
        flat_updates = torch.stack([self._flatten_weights(w) for w in malicious_local_updates])
        mu = torch.mean(flat_updates, dim=0)
        std = torch.std(flat_updates, dim=0) + 1e-9

        # 2. 均值反方向偏移
        direction = torch.sign(mu)

        # 3. 将恶意参数推到修剪边界
        best_malicious_flat = mu - self.b * std * direction

        return self._unflatten_weights(best_malicious_flat, malicious_local_updates[0])


class KrumAttack:
    """
    专门针对 Krum 聚合规则的局部模型投毒攻击
    """

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
        # 1. 展平并计算参考分布
        flat_updates = torch.stack([self._flatten_weights(w) for w in malicious_local_updates])
        mu = torch.mean(flat_updates, dim=0)
        std = torch.std(flat_updates, dim=0) + 1e-9
        direction = torch.sign(mu)

        # 2. 初始化二分查找参数
        gamma_succ = 0.0
        gamma_fail = 3.0
        gamma = gamma_fail / 2.0
        threshold = 1e-4

        # Krum 算法选择距离最近的邻居数量
        krum_k = max(1, self.num_total - self.num_attackers - 2)

        def calc_krum_score(target_vec, pool_benign, cand_vec, num_mal):
            dists_to_benign = torch.cdist(target_vec.unsqueeze(0), pool_benign)[0]
            dists_to_mal = torch.cdist(target_vec.unsqueeze(0), cand_vec.unsqueeze(0))[0].repeat(num_mal)
            all_dists = torch.cat([dists_to_benign, dists_to_mal])
            all_dists, _ = torch.sort(all_dists)
            return torch.sum(all_dists[1:krum_k + 1])

        # 3. 二分查找寻找最优 gamma
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

        # 4. 生成最终权重
        best_malicious_flat = mu - gamma_succ * std * direction
        return self._unflatten_weights(best_malicious_flat, malicious_local_updates[0])


import torch
import copy


class CerPAttack:
    """
    CerP: Stealthy and Colluded Backdoor Attack against Federated Learning (AAAI 2023)
    核心机制: 分布式触发器切分 + 隐蔽性模型约束
    """

    def __init__(self, config):
        self.config = config
        self.name = 'CerPAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.adversarial_clients = config.get('apply_to_client_ids', [])

        # 确定当前恶意客户端在所有恶意客户端中的索引
        if self.client_id in self.adversarial_clients:
            self.attacker_idx = self.adversarial_clients.index(self.client_id)
        else:
            self.attacker_idx = 0

        self.num_attackers = max(1, len(self.adversarial_clients))
        self.trigger_size = config.get('trigger_width', 4)  # 假设为正方形触发器
        self.epsilon = config.get('epsilon', 0.5)  # 隐蔽性约束的 L2 范数阈值

    def should_apply(self, round_idx):
        """判断当前轮次是否触发攻击"""
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def poison_data(self, data, target):
        """
        数据投毒：CerP 核心逻辑 1 - 分布式触发器
        将一个 trigger 划分为多个 sub-trigger，交由不同的 client 执行。
        为了简单起见，我们将右下角的触发器区域按 attacker_idx 分配不同的像素点或小块。
        """
        poisoned_data = data.clone()
        poisoned_target = target.clone()

        batch_size = data.size(0)
        channels, height, width = data.size()[1:]

        # 定义全局触发器区域 (右下角)
        start_h = height - self.trigger_size
        start_w = width - self.trigger_size

        # 简单的切分逻辑：将触发器区域横向切分为多个小条块，每个 attacker 负责一条
        # 这样组合起来才是一个完整的特征
        chunk_size = max(1, self.trigger_size // self.num_attackers)
        my_chunk_start = self.attacker_idx * chunk_size
        my_chunk_end = my_chunk_start + chunk_size if self.attacker_idx < self.num_attackers - 1 else self.trigger_size

        # 只有特定的后门比例才会被投毒
        poison_ratio = self.config.get('poison_ratio', 0.5)
        num_poisoned = int(batch_size * poison_ratio)

        if num_poisoned > 0:
            # 修改标签
            poisoned_target[:num_poisoned] = self.target_class

            # 添加属于该 attacker 的局部触发器 (最高亮度 / 特定像素值)
            # 在标准化后的图像中，2.5 通常代表接近全白的像素
            for i in range(num_poisoned):
                for c in range(channels):
                    poisoned_data[i, c,
                    start_h + my_chunk_start: start_h + my_chunk_end,
                    start_w: start_w + self.trigger_size] = 2.5

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        """
        模型投毒：CerP 核心逻辑 2 - 隐蔽性约束 (Stealthy Tuning)
        通过投影 (Projection) 将本地模型的更新限制在全局模型的一个安全半径 (epsilon) 内，
        从而骗过 Krum, Trimmed Mean 等鲁棒聚合算法。
        """
        poisoned_state = {}

        # 计算当前本地模型与全局模型的差值 (更新量)
        update_vector = []
        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            update_vector.append(diff.view(-1))

        update_vector = torch.cat(update_vector)
        l2_norm = torch.norm(update_vector, p=2)

        # 如果更新幅度超过了安全阈值 epsilon，则进行缩放投影 (Clipping)
        if l2_norm > self.epsilon:
            scale_factor = self.epsilon / (l2_norm + 1e-9)
            print(f"   [CerP] Client {self.client_id} clipping model update (Norm {l2_norm:.2f} -> {self.epsilon})")

            for k in local_model_state.keys():
                diff = local_model_state[k] - global_model_state[k]
                poisoned_state[k] = global_model_state[k] + diff * scale_factor
        else:
            poisoned_state = copy.deepcopy(local_model_state)

        return poisoned_state


import torch
import torch.nn as nn


class A3FLAttack:
    """
    A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning
    核心机制：在本地训练前，通过对抗性梯度更新，动态优化出一个具有抗遗忘能力的触发器补丁。
    """

    def __init__(self, config):
        self.config = config
        self.name = 'A3FLAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])

        self.trigger_height = config.get('trigger_height', 5)
        self.trigger_width = config.get('trigger_width', 5)

        # 初始化一个可优化的触发器补丁 (Trigger Patch)
        self.trigger_patch = None

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def train_attack_model(self, model, dataloader, client_id, device, verbose=False):
        """
        利用您框架中的钩子，在本地正常训练前，先执行 A3FL 的触发器对抗优化。
        """
        if client_id not in self.apply_to_client_ids:
            return

        print(f"   [A3FL] Optimizing Adversarially Adaptive Trigger for client {client_id}...")

        # 冻结模型参数，只优化触发器补丁
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if self.trigger_patch is None:
            # 假设输入通道为3 (CIFAR-10)
            self.trigger_patch = torch.zeros((3, self.trigger_height, self.trigger_width), device=device,
                                             requires_grad=True)
        else:
            self.trigger_patch = self.trigger_patch.detach().clone().to(device)
            self.trigger_patch.requires_grad = True

        optimizer_trigger = torch.optim.Adam([self.trigger_patch], lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # A3FL 的代理对抗优化：寻找最能激活后门目标的 Trigger 扰动
        # 在完整论文中，这里包含针对 unlearning 的 minimax 优化
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

                # 将当前正在优化的补丁叠加到图像上
                poisoned_data[:, :, start_h:, start_w:] = poisoned_data[:, :, start_h:, start_w:] + self.trigger_patch

                # 伪造全为目标类的标签
                poisoned_target = torch.ones(num_poison, dtype=torch.long, device=device) * self.target_class

                optimizer_trigger.zero_grad()
                output = model(poisoned_data)

                # 优化目标：使得带触发器的图像被识别为目标类的置信度最大化
                loss = criterion(output, poisoned_target)
                loss.backward()
                optimizer_trigger.step()

                # 对触发器进行 L_inf 约束，保证隐蔽性 (假设图像被标准化，这里的扰动约束设为 0.5)
                with torch.no_grad():
                    self.trigger_patch.clamp_(-0.5, 0.5)

        self.trigger_patch.requires_grad = False

        # 恢复模型为训练状态，供后续的 FLClient.train 正常使用
        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def poison_data(self, data, target):
        """在实际的数据加载循环中，贴上刚刚优化好的自适应触发器"""
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

            # 叠加自适应触发器
            poisoned_data[:num_poisoned, :, start_h:, start_w:] = poisoned_data[:num_poisoned, :, start_h:,
                                                                  start_w:] + optimized_patch

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        """A3FL 是自适应触发器攻击，可选择结合模型缩放 (Scaling) 增加破坏力"""
        scaling_factor = self.config.get('scaling_factor', 1)
        if scaling_factor > 1:
            poisoned_state = {}
            for k in local_model_state.keys():
                diff = local_model_state[k] - global_model_state[k]
                poisoned_state[k] = global_model_state[k] + diff * scaling_factor
            return poisoned_state
        return local_model_state


import torch
import copy


class FCBAAttack:
    """
    FCBA: Full Combination Backdoor Attack (AAAI 2024)
    核心机制：通过组合多个触发器特征（如多位置、多通道模式），构建更完整、更难被良性更新洗掉的持久性后门。
    """

    def __init__(self, config):
        self.config = config
        self.name = 'FCBAAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])

        # FCBA 组合触发器的基础大小
        self.trigger_size = config.get('trigger_width', 4)
        self.scaling_factor = config.get('scaling_factor', 1)

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def poison_data(self, data, target):
        """
        数据投毒：FCBA 核心逻辑 - 植入全组合触发器 (Full Combination Trigger)
        为了实现组合特征，我们在图像的多个关键位置（如四个角）同时植入触发器，
        或者使用特定的强通道组合，使模型必须学习这个"完整模式"才能触发后门。
        """
        poisoned_data = data.clone()
        poisoned_target = target.clone()

        batch_size = data.size(0)
        channels, height, width = data.size()[1:]

        poison_ratio = self.config.get('poison_ratio', 0.5)
        num_poisoned = int(batch_size * poison_ratio)

        if num_poisoned > 0:
            # 修改标签为目标类别
            poisoned_target[:num_poisoned] = self.target_class

            # 植入组合触发器：在四个角同时添加特征块，形成复杂的全局模式
            # (相较于单一角落的 BadNets，这种组合模式在卷积网络中更难被遗忘)
            pixel_val = 2.5  # 标准化后的高亮像素

            # 左上角 (Top-Left)
            poisoned_data[:num_poisoned, :, 0:self.trigger_size, 0:self.trigger_size] = pixel_val
            # 右上角 (Top-Right)
            poisoned_data[:num_poisoned, :, 0:self.trigger_size, width - self.trigger_size:width] = pixel_val
            # 左下角 (Bottom-Left)
            poisoned_data[:num_poisoned, :, height - self.trigger_size:height, 0:self.trigger_size] = pixel_val
            # 右下角 (Bottom-Right)
            poisoned_data[:num_poisoned, :, height - self.trigger_size:height,
            width - self.trigger_size:width] = pixel_val

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        """
        模型投毒：为了确保这些组合特征能在全局模型中占据主导地位，
        结合一定的 Scaling 放大恶意权重更新。
        """
        if self.scaling_factor <= 1:
            return local_model_state

        poisoned_state = {}
        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            poisoned_state[k] = global_model_state[k] + diff * self.scaling_factor

        return poisoned_state


import torch
import torch.nn as nn
import copy


class IBAGenerator(nn.Module):
    """
    用于生成 IBA 实例相关隐蔽触发器的轻量级生成器
    """

    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 确保输出范围在 [-1, 1] 之间，便于控制 L_inf 范数
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class IBAAttack:
    """
    IBA: Towards Irreversible Backdoor Attacks in Federated Learning (NeurIPS 2023)
    核心机制: 生成式隐蔽触发器 (Trigger Generating) + PGD受约模型投毒 (Partial Model Poisoning)
    """

    def __init__(self, config):
        self.config = config
        self.name = 'IBAAttack'
        self.target_class = config.get('target_class', 0)
        self.client_id = config.get('client_id')
        self.apply_to_client_ids = config.get('apply_to_client_ids', [])

        self.epsilon = config.get('epsilon', 0.1)  # 视觉隐蔽性 L_inf 阈值
        self.pgd_bound = config.get('pgd_bound', 2.0)  # PGD 模型空间约束半径
        self.scaling_factor = config.get('scaling_factor', 1)  # Model Replacement 放缩倍数

        self.generator = None

    def should_apply(self, round_idx):
        start = self.config.get('attack_start_round', 0)
        stop = self.config.get('attack_stop_round', 1000)
        freq = self.config.get('attack_frequency', 1)
        return (round_idx >= start) and (round_idx < stop) and ((round_idx - start) % freq == 0)

    def train_attack_model(self, model, dataloader, client_id, device, verbose=False):
        """阶段 1：利用全局模型训练本地的触发器生成器"""
        if client_id not in self.apply_to_client_ids:
            return

        print(f"   [IBA] Training instance-specific trigger generator for client {client_id}...")

        # 冻结当前的全局分类模型参数
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # 初始化生成器
        if self.generator is None:
            # 动态获取图像通道数
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

                # 生成对抗噪声并叠加 (噪声被严格约束在 [-epsilon, epsilon] 内)
                noise = self.generator(data) * self.epsilon
                poisoned_data = data + noise

                # 优化目标：使得带噪声的图像被全局模型误判为 target_class
                poisoned_target = torch.ones(batch_size, dtype=torch.long, device=device) * self.target_class

                optimizer_g.zero_grad()
                output = model(poisoned_data)
                loss = criterion(output, poisoned_target)
                loss.backward()
                optimizer_g.step()

        self.generator.eval()

        # 恢复分类模型状态，供主框架后续的本地训练使用
        for param in model.parameters():
            param.requires_grad = True
        model.train()

    def poison_data(self, data, target):
        """阶段 2 (数据层)：应用刚刚训练好的生成器进行数据投毒"""
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

                # 叠加对抗触发器
                noise = self.generator(clean_subset) * self.epsilon
                poisoned_subset = clean_subset + noise
                poisoned_data[:num_poisoned] = poisoned_subset

        return poisoned_data, poisoned_target

    def apply_model_poisoning(self, local_model_state, global_model_state, algorithm):
        """阶段 2 (模型层)：实施 PGD 空间约束与 Model Replacement 放缩"""
        poisoned_state = {}

        # 1. 计算本次的更新差值向量
        update_vector = []
        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            update_vector.append(diff.view(-1))

        update_vector = torch.cat(update_vector)
        l2_norm = torch.norm(update_vector, p=2)

        # 2. 如果更新幅度超出了安全半径，进行裁剪投影 (PGD)
        scale_factor = 1.0
        if l2_norm > self.pgd_bound:
            scale_factor = self.pgd_bound / (l2_norm + 1e-9)
            print(f"   [IBA] Applying PGD constraint: L2-Norm clipped from {l2_norm:.2f} down to {self.pgd_bound}")

        # 3. 结合 Model Replacement (MR) 倍数进行最终调整
        final_scale = scale_factor * self.scaling_factor

        for k in local_model_state.keys():
            diff = local_model_state[k] - global_model_state[k]
            poisoned_state[k] = global_model_state[k] + diff * final_scale

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
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
