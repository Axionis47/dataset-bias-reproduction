"""
Training loop for dataset bias reproduction.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_data_loaders, MixUpCutMix
from src.models.model_factory import (
    create_model_from_config,
    create_loss_function,
    create_optimizer,
    create_scheduler,
    WarmupScheduler,
)

console = Console()
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for dataset classification."""
    
    def __init__(self, config: Dict, experiment_dir: Path):
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = self._setup_device()
        
        # Create directories
        self.checkpoints_dir = experiment_dir / "checkpoints"
        self.logs_dir = experiment_dir / "logs"
        self.results_dir = experiment_dir / "results"
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Reproducibility
        self._setup_reproducibility()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None
        self.mixup_cutmix = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_metrics = []
        self.val_metrics = []
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (MPS/CPU)."""
        hardware_config = self.config['hardware']
        device_setting = hardware_config.get('device', 'auto')
        
        if device_setting == 'auto':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                console.print("✅ Using Apple Silicon MPS")
            else:
                device = torch.device('cpu')
                console.print("⚠️  MPS not available, using CPU")
        else:
            device = torch.device(device_setting)
            console.print(f"Using specified device: {device}")
        
        return device
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / "training.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )
    
    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        repro_config = self.config['reproducibility']
        seed = repro_config['seed']
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        if repro_config.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set random seed to {seed}")
    
    def setup_model_and_training(self, datasets: List[str]):
        """Setup model, optimizer, scheduler, and loss function."""
        # Update num_classes in config
        self.config['model']['num_classes'] = len(datasets)
        
        # Create model
        self.model = create_model_from_config(self.config)
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = create_optimizer(self.model, self.config)
        
        # Create scheduler with warmup
        base_scheduler = create_scheduler(self.optimizer, self.config)
        warmup_epochs = self.config['training'].get('warmup_epochs', 5)
        
        if warmup_epochs > 0:
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                base_scheduler=base_scheduler,
            )
        else:
            self.scheduler = base_scheduler
        
        # Create loss function
        self.loss_fn = create_loss_function(self.config)
        
        # Setup mixed precision
        if self.config['training'].get('use_amp', True) and self.device.type != 'cpu':
            self.scaler = GradScaler()
        
        # Setup MixUp/CutMix
        mixup_config = self.config['data']['train_transform'].get('mixup', {})
        cutmix_config = self.config['data']['train_transform'].get('cutmix', {})
        
        if mixup_config or cutmix_config:
            self.mixup_cutmix = MixUpCutMix(
                mixup_alpha=mixup_config.get('alpha', 0.8),
                cutmix_alpha=cutmix_config.get('alpha', 1.0),
                num_classes=len(datasets),
            )
        
        logger.info(f"Model: {self.config['model']['architecture']}")
        logger.info(f"Optimizer: {self.config['training']['optimizer']}")
        logger.info(f"Scheduler: {self.config['training']['scheduler']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.scaler is not None}")
        logger.info(f"MixUp/CutMix: {self.mixup_cutmix is not None}")
    
    def train_epoch(self, train_loader) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task(
                f"[green]Epoch {self.current_epoch + 1} Training",
                total=len(train_loader)
            )
            
            for batch_idx, (images, labels, metadata) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Apply MixUp/CutMix if configured
                if self.mixup_cutmix:
                    images, labels = self.mixup_cutmix((images, labels))
                
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracy (handle mixup labels)
                if labels.dim() == 1:
                    # Standard labels
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                else:
                    # Mixed labels - use argmax of soft targets
                    _, predicted = outputs.max(1)
                    _, target_labels = labels.max(1)
                    correct += predicted.eq(target_labels).sum().item()
                
                total += labels.size(0)
                
                progress.advance(task)
                
                # Log every N batches
                if batch_idx % self.config['logging']['log_every'] == 0:
                    current_acc = 100.0 * correct / total
                    current_loss = total_loss / (batch_idx + 1)
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}: "
                              f"Loss={current_loss:.4f}, Acc={current_acc:.2f}%")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr'],
        }
    
    def validate_epoch(self, val_loader) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                
                task = progress.add_task(
                    f"[blue]Epoch {self.current_epoch + 1} Validation",
                    total=len(val_loader)
                )
                
                for images, labels, metadata in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    if self.scaler:
                        with autocast():
                            outputs = self.model(images)
                            loss = self.loss_fn(outputs, labels)
                    else:
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                    
                    # Update metrics
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                    
                    progress.advance(task)
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoints_dir / "latest.ckpt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best.ckpt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with validation accuracy: {self.best_val_acc:.2f}%")
    
    def train(self, train_loader, val_loader) -> Dict:
        """Run complete training loop."""
        console.print(f"\n[bold green]Starting training for {self.config['training']['epochs']} epochs[/bold green]")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            self.train_metrics.append(train_metrics)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(epoch)
            
            # Check if best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
            
            # Save checkpoint
            save_every = self.config['training'].get('save_every', 5)
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Log epoch results
            console.print(f"\n[bold]Epoch {epoch + 1}/{self.config['training']['epochs']}[/bold]")
            console.print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            console.print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            console.print(f"Learning Rate: {train_metrics['learning_rate']:.6f}")
            if is_best:
                console.print("[green]✅ New best model![/green]")
        
        # Training complete
        total_time = time.time() - start_time
        
        # Save final metrics
        final_metrics = {
            'experiment': self.config['experiment']['name'],
            'total_epochs': self.config['training']['epochs'],
            'best_val_accuracy': self.best_val_acc,
            'final_train_accuracy': self.train_metrics[-1]['accuracy'],
            'final_val_accuracy': self.val_metrics[-1]['accuracy'],
            'total_training_time': total_time,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        metrics_file = self.results_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        console.print(f"\n[bold green]Training Complete![/bold green]")
        console.print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        console.print(f"Total training time: {total_time/3600:.2f} hours")
        console.print(f"Metrics saved to: {metrics_file}")
        
        return final_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train dataset classification model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model architecture"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="~/dataset_bias_data",
        help="Base directory for data"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.model:
        config['model']['architecture'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs

    # Create experiment directory
    experiment_name = config['experiment']['name']
    experiment_dir = Path("experiments") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config to experiment directory
    config_file = experiment_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    try:
        # Create trainer
        trainer = Trainer(config, experiment_dir)

        # Create data loaders
        datasets_to_use = config['experiment']['datasets']
        train_loader, val_loader, available_datasets = create_data_loaders(
            config,
            base_dir=args.base_dir,
            dataset_filter=datasets_to_use,
        )

        console.print(f"Available datasets: {available_datasets}")
        console.print(f"Training batches: {len(train_loader)}")
        console.print(f"Validation batches: {len(val_loader)}")

        # Setup model and training components
        trainer.setup_model_and_training(available_datasets)

        # Run training
        results = trainer.train(train_loader, val_loader)

        console.print("[bold green]Training completed successfully![/bold green]")
        return 0

    except KeyboardInterrupt:
        console.print("\n[red]Training interrupted by user[/red]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]")
        logger.exception("Training failed")
        return 1


if __name__ == "__main__":
    exit(main())
