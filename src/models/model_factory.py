"""
Model factory for dataset bias reproduction.
Provides various architectures tested in the paper.
"""

import logging
from typing import Dict, Optional

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating models used in dataset bias experiments."""
    
    SUPPORTED_MODELS = {
        'alexnet': 'legacy_alexnet.in1k',
        'vgg16': 'vgg16',
        'resnet50': 'resnet50',
        'vit_small': 'vit_small_patch16_224',
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'convnext_base': 'convnext_base',
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> nn.Module:
        """
        Create a model for dataset classification.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of dataset classes
            pretrained: Whether to use pretrained weights
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate (for transformers)
            
        Returns:
            PyTorch model
        """
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Supported models: {list(cls.SUPPORTED_MODELS.keys())}")
        
        timm_name = cls.SUPPORTED_MODELS[model_name]
        
        logger.info(f"Creating {model_name} ({timm_name}) with {num_classes} classes")

        # Handle special cases
        if model_name == 'alexnet':
            # Use torchvision AlexNet
            from torchvision.models import alexnet
            model = alexnet(pretrained=pretrained)
            # Replace classifier for our number of classes
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        else:
            # Create model using timm
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {total_params:,} total parameters, "
                   f"{trainable_params:,} trainable")
        
        return model
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get information about a model architecture."""
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Model information from the paper
        model_info = {
            'alexnet': {
                'paper_accuracy': 77.8,  # YCD combination
                'parameters': '60M',
                'year': 2012,
                'description': 'Classic CNN architecture',
            },
            'vgg16': {
                'paper_accuracy': 83.5,
                'parameters': '138M',
                'year': 2014,
                'description': 'Deep CNN with small filters',
            },
            'resnet50': {
                'paper_accuracy': 83.8,
                'parameters': '25M',
                'year': 2015,
                'description': 'Residual network',
            },
            'vit_small': {
                'paper_accuracy': 82.4,
                'parameters': '22M',
                'year': 2020,
                'description': 'Vision Transformer',
            },
            'convnext_tiny': {
                'paper_accuracy': 84.7,
                'parameters': '27M',
                'year': 2022,
                'description': 'Modern ConvNet (default choice)',
            },
            'convnext_small': {
                'paper_accuracy': None,
                'parameters': '50M',
                'year': 2022,
                'description': 'Larger ConvNeXt variant',
            },
            'convnext_base': {
                'paper_accuracy': None,
                'parameters': '89M',
                'year': 2022,
                'description': 'Base ConvNeXt variant',
            },
        }
        
        return model_info.get(model_name, {})
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """List all supported models with their information."""
        models_info = {}
        for model_name in cls.SUPPORTED_MODELS:
            models_info[model_name] = cls.get_model_info(model_name)
        return models_info


def create_model_from_config(config: Dict) -> nn.Module:
    """Create model from configuration."""
    model_config = config['model']
    
    model = ModelFactory.create_model(
        model_name=model_config['architecture'],
        num_classes=model_config['num_classes'],
        pretrained=model_config.get('pretrained', False),
        drop_rate=model_config.get('drop_rate', 0.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.0),
    )
    
    return model


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred: Predictions [batch_size, num_classes]
            target: Targets [batch_size] or [batch_size, num_classes] for mixup
        """
        if target.dim() == 1:
            # Standard labels
            num_classes = pred.size(-1)
            target_onehot = torch.zeros_like(pred)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)
            
            # Apply label smoothing
            target_smooth = target_onehot * (1 - self.smoothing) + self.smoothing / num_classes
        else:
            # Already one-hot (from mixup/cutmix)
            target_smooth = target
        
        # Compute cross entropy with soft targets
        log_pred = torch.log_softmax(pred, dim=-1)
        loss = -torch.sum(target_smooth * log_pred, dim=-1)
        
        return loss.mean()


def create_loss_function(config: Dict) -> nn.Module:
    """Create loss function from configuration."""
    training_config = config['training']
    
    label_smoothing = training_config.get('label_smoothing', 0.0)
    
    if label_smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        return nn.CrossEntropyLoss()


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    training_config = config['training']
    
    optimizer_name = training_config.get('optimizer', 'adamw').lower()
    lr = training_config.get('learning_rate', 1e-3)
    weight_decay = training_config.get('weight_decay', 0.3)
    
    if optimizer_name == 'adamw':
        betas = training_config.get('betas', [0.9, 0.95])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif optimizer_name == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration."""
    training_config = config['training']
    
    scheduler_name = training_config.get('scheduler', 'cosine').lower()
    
    if scheduler_name == 'cosine':
        epochs = training_config.get('epochs', 30)
        warmup_epochs = training_config.get('warmup_epochs', 5)
        
        # Simple cosine annealing (warmup handled separately)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=0,
        )
        return scheduler
    elif scheduler_name == 'step':
        step_size = training_config.get('step_size', 10)
        gamma = training_config.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        return scheduler
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class WarmupScheduler:
    """Warmup scheduler wrapper."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.base_scheduler:
            # Use base scheduler after warmup
            self.base_scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
