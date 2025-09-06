"""
Dataset classes for dataset bias reproduction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class DatasetBiasDataset(Dataset):
    """Dataset for dataset classification task."""

    def __init__(
        self,
        manifest_path: str,
        base_dir: str = "~/dataset_bias_data",
        transform: Optional[transforms.Compose] = None,
        dataset_filter: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            manifest_path: Path to CSV manifest file
            base_dir: Base directory for data
            transform: Image transforms to apply
            dataset_filter: List of datasets to include (None = all)
        """
        self.base_dir = Path(base_dir).expanduser()
        self.transform = transform
        
        # Load manifest
        self.manifest = pd.read_csv(manifest_path)
        
        # Filter datasets if specified
        if dataset_filter:
            self.manifest = self.manifest[
                self.manifest['dataset'].isin(dataset_filter)
            ].reset_index(drop=True)
        
        # Create dataset to class mapping
        self.datasets = sorted(self.manifest['dataset'].unique())
        self.dataset_to_class = {dataset: i for i, dataset in enumerate(self.datasets)}
        self.class_to_dataset = {i: dataset for dataset, i in self.dataset_to_class.items()}
        
        logger.info(f"Loaded {len(self.manifest)} samples from {len(self.datasets)} datasets")
        logger.info(f"Datasets: {self.datasets}")
        
        # Log class distribution
        for dataset in self.datasets:
            count = len(self.manifest[self.manifest['dataset'] == dataset])
            logger.info(f"  {dataset}: {count} samples (class {self.dataset_to_class[dataset]})")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get item by index.
        
        Returns:
            image: Transformed image tensor
            label: Dataset class label (0, 1, 2, ...)
            metadata: Additional information
        """
        row = self.manifest.iloc[idx]
        
        # Load image
        image_path = self.base_dir / row['final_path']
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply transforms
                if self.transform:
                    img = self.transform(img)
                
                # Get label
                label = self.dataset_to_class[row['dataset']]
                
                # Metadata
                metadata = {
                    'dataset': row['dataset'],
                    'image_path': str(image_path),
                    'width': row.get('width', 0),
                    'height': row.get('height', 0),
                    'sha256': row.get('sha256', ''),
                }
                
                return img, label, metadata
                
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback_img = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                fallback_img = transforms.ToTensor()(Image.new('RGB', (224, 224), (0, 0, 0)))
            
            label = self.dataset_to_class[row['dataset']]
            metadata = {'dataset': row['dataset'], 'error': str(e)}
            
            return fallback_img, label, metadata

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        class_counts = self.manifest['dataset'].value_counts()
        total_samples = len(self.manifest)
        
        weights = []
        for dataset in self.datasets:
            count = class_counts.get(dataset, 1)
            weight = total_samples / (len(self.datasets) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


def get_train_transforms(config: Dict) -> transforms.Compose:
    """Get training transforms based on config."""
    transform_list = []
    
    # Resize and crop
    resize_crop = config.get('resize_crop', 224)
    transform_list.extend([
        transforms.RandomResizedCrop(
            resize_crop,
            scale=(0.08, 1.0),
            ratio=(3./4., 4./3.),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    # RandAugment
    randaug_config = config.get('randaug', {})
    if randaug_config:
        from torchvision.transforms import RandAugment
        magnitude = randaug_config.get('magnitude', 9)
        num_ops = randaug_config.get('num_ops', 2)
        transform_list.append(
            RandAugment(num_ops=num_ops, magnitude=magnitude)
        )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict) -> transforms.Compose:
    """Get validation transforms based on config."""
    resize = config.get('resize', 256)
    center_crop = config.get('center_crop', 224)
    
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class MixUpCutMix:
    """MixUp and CutMix augmentation."""
    
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        num_classes: int = 3,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp or CutMix to batch."""
        images, labels = batch
        
        if torch.rand(1) > self.prob:
            return images, labels
        
        # Convert labels to one-hot
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        if torch.rand(1) < self.switch_prob:
            # MixUp
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            batch_size = images.size(0)
            index = torch.randperm(batch_size, device=images.device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
            
            return mixed_images, mixed_labels
        else:
            # CutMix
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
            batch_size = images.size(0)
            index = torch.randperm(batch_size, device=images.device)
            
            # Get random box
            W, H = images.size(-1), images.size(-2)
            cut_rat = torch.sqrt(1. - lam)
            cut_w = (W * cut_rat).int()
            cut_h = (H * cut_rat).int()
            
            cx = torch.randint(W, (1,), device=images.device)
            cy = torch.randint(H, (1,), device=images.device)
            
            bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
            bby1 = torch.clamp(cy - cut_h // 2, 0, H)
            bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
            bby2 = torch.clamp(cy + cut_h // 2, 0, H)
            
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda based on actual cut area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
            
            return mixed_images, mixed_labels


def create_data_loaders(
    config: Dict,
    base_dir: str = "~/dataset_bias_data",
    dataset_filter: Optional[List[str]] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
    """Create train and validation data loaders."""
    
    base_path = Path(base_dir).expanduser()
    train_manifest = base_path / "sampled" / "train_manifest.csv"
    val_manifest = base_path / "sampled" / "val_manifest.csv"
    
    # Get transforms
    train_transform = get_train_transforms(config['data']['train_transform'])
    val_transform = get_val_transforms(config['data']['val_transform'])
    
    # Create datasets
    train_dataset = DatasetBiasDataset(
        manifest_path=str(train_manifest),
        base_dir=base_dir,
        transform=train_transform,
        dataset_filter=dataset_filter,
    )
    
    val_dataset = DatasetBiasDataset(
        manifest_path=str(val_manifest),
        base_dir=base_dir,
        transform=val_transform,
        dataset_filter=dataset_filter,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=config['hardware']['persistent_workers'],
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=config['hardware']['persistent_workers'],
        drop_last=False,
    )
    
    return train_loader, val_loader, train_dataset.datasets
