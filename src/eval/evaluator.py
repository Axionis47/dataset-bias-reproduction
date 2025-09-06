"""
Evaluation system for dataset bias reproduction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from sklearn.metrics import confusion_matrix, classification_report
from torch.cuda.amp import autocast

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_data_loaders
from src.models.model_factory import create_model_from_config

console = Console()
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for dataset classification models."""
    
    def __init__(self, config: Dict, checkpoint_path: str, experiment_dir: Path):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.experiment_dir = experiment_dir
        self.device = self._setup_device()
        
        # Create results directory
        self.results_dir = experiment_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and data
        self.model = None
        self.datasets = None
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def load_model(self, datasets: List[str]) -> nn.Module:
        """Load trained model from checkpoint."""
        console.print(f"[yellow]Loading model from {self.checkpoint_path}[/yellow]")
        
        # Update config with correct number of classes
        self.config['model']['num_classes'] = len(datasets)
        
        # Create model
        model = create_model_from_config(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        console.print(f"✅ Model loaded successfully")
        console.print(f"Best validation accuracy from training: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        
        return model
    
    def evaluate_model(self, model: nn.Module, data_loader, dataset_names: List[str]) -> Dict:
        """Evaluate model on data loader."""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                
                task = progress.add_task("[blue]Evaluating", total=len(data_loader))
                
                for batch_idx, (images, labels, metadata) in enumerate(data_loader):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    if self.device.type != 'cpu':
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    # Get predictions and probabilities
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    # Store results
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    # Update metrics
                    total_loss += loss.item()
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                    
                    progress.advance(task)
        
        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(data_loader)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i, dataset_name in enumerate(dataset_names):
            class_mask = all_labels == i
            if class_mask.sum() > 0:
                class_correct = (all_predictions[class_mask] == all_labels[class_mask]).sum()
                class_total = class_mask.sum()
                per_class_accuracy[dataset_name] = 100.0 * class_correct / class_total
            else:
                per_class_accuracy[dataset_name] = 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'per_class_accuracy': per_class_accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'total_samples': total,
        }
    
    def create_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, 
                              dataset_names: List[str], save_path: Path) -> None:
        """Create and save confusion matrix."""
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=dataset_names, yticklabels=dataset_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted Dataset')
        ax1.set_ylabel('True Dataset')
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=dataset_names, yticklabels=dataset_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Dataset')
        ax2.set_ylabel('True Dataset')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"✅ Confusion matrix saved to {save_path}")
    
    def create_classification_report(self, labels: np.ndarray, predictions: np.ndarray,
                                   dataset_names: List[str]) -> Dict:
        """Create detailed classification report."""
        report = classification_report(
            labels, predictions, 
            target_names=dataset_names,
            output_dict=True
        )
        return report
    
    def run_evaluation(self, base_dir: str = "~/dataset_bias_data") -> Dict:
        """Run complete evaluation."""
        console.print(f"\n[bold green]Starting Evaluation[/bold green]")
        
        # Create data loaders
        datasets_to_use = self.config['experiment']['datasets']
        _, val_loader, available_datasets = create_data_loaders(
            self.config,
            base_dir=base_dir,
            dataset_filter=datasets_to_use,
        )
        
        console.print(f"Evaluating on datasets: {available_datasets}")
        console.print(f"Validation batches: {len(val_loader)}")
        
        # Load model
        model = self.load_model(available_datasets)
        
        # Evaluate
        results = self.evaluate_model(model, val_loader, available_datasets)
        
        # Create visualizations
        cm_path = self.results_dir / "confusion_matrix.png"
        self.create_confusion_matrix(
            results['labels'], 
            results['predictions'], 
            available_datasets, 
            cm_path
        )
        
        # Classification report
        classification_report_dict = self.create_classification_report(
            results['labels'],
            results['predictions'],
            available_datasets
        )
        
        # Compile final results
        final_results = {
            'experiment': self.config['experiment']['name'],
            'datasets': available_datasets,
            'num_classes': len(available_datasets),
            'total_samples': results['total_samples'],
            'overall_accuracy': results['accuracy'],
            'overall_loss': results['loss'],
            'per_class_accuracy': results['per_class_accuracy'],
            'classification_report': classification_report_dict,
            'chance_accuracy': 100.0 / len(available_datasets),
        }
        
        # Save results
        results_file = self.results_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        console.print(f"\n[bold green]Evaluation Complete![/bold green]")
        console.print(f"Overall Accuracy: {results['accuracy']:.2f}%")
        console.print(f"Chance Accuracy: {100.0 / len(available_datasets):.2f}%")
        console.print(f"Improvement over chance: {results['accuracy'] - 100.0 / len(available_datasets):.2f}%")
        
        console.print(f"\nPer-class Accuracy:")
        for dataset, acc in results['per_class_accuracy'].items():
            console.print(f"  {dataset}: {acc:.2f}%")
        
        console.print(f"\nResults saved to: {results_file}")
        console.print(f"Confusion matrix: {cm_path}")
        
        return final_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate dataset classification model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
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
    
    # Get experiment directory
    experiment_name = config['experiment']['name']
    experiment_dir = Path("experiments") / experiment_name
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create evaluator
        evaluator = Evaluator(config, args.checkpoint, experiment_dir)
        
        # Run evaluation
        results = evaluator.run_evaluation(base_dir=args.base_dir)
        
        console.print("[bold green]Evaluation completed successfully![/bold green]")
        return 0
        
    except Exception as e:
        console.print(f"\n[red]Evaluation failed: {e}[/red]")
        logger.exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    exit(main())
