"""
Dataset sampler for dataset bias reproduction.
Implements uniform sampling per dataset with deduplication.
"""

import argparse
import hashlib
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import imagehash
import pandas as pd
from PIL import Image
from rich.console import Console
from rich.progress import track
from sklearn.model_selection import train_test_split

console = Console()
logger = logging.getLogger(__name__)


class DatasetSampler:
    """Samples and preprocesses datasets for training."""

    def __init__(
        self,
        base_dir: str = "~/dataset_bias_data",
        n_per_dataset: int = 10000,
        val_size: int = 1000,
        seed: int = 42,
    ):
        self.base_dir = Path(base_dir).expanduser()
        self.n_per_dataset = n_per_dataset
        self.val_size = val_size
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        
        # Directories
        self.metadata_dir = self.base_dir / "metadata"
        self.raw_dir = self.base_dir / "raw"
        self.sampled_dir = self.base_dir / "sampled"
        
        # Create sampled directory
        self.sampled_dir.mkdir(parents=True, exist_ok=True)
        
        # Available datasets
        self.available_datasets = self._find_available_datasets()

    def _find_available_datasets(self) -> List[str]:
        """Find datasets that have been successfully downloaded."""
        available = []
        
        for dataset_name in ['yfcc', 'cc', 'datacomp', 'wit', 'laion', 'imagenet']:
            index_file = self.metadata_dir / f"{dataset_name}_index.csv"
            skipped_file = self.metadata_dir / f"{dataset_name}_skipped.txt"
            
            if index_file.exists():
                # Check if we have successful downloads
                try:
                    df = pd.read_csv(index_file)
                    successful = df[df['failure_reason'].isna()]
                    if len(successful) > 0:
                        available.append(dataset_name)
                        console.print(f"✅ Found {len(successful)} images for {dataset_name}")
                    else:
                        console.print(f"⚠️  No successful downloads for {dataset_name}")
                except Exception as e:
                    console.print(f"❌ Error reading {dataset_name} index: {e}")
            elif skipped_file.exists():
                console.print(f"⏭️  {dataset_name} was skipped")
            else:
                console.print(f"❌ No data found for {dataset_name}")
        
        return available

    def _compute_perceptual_hash(self, image_path: Path) -> str:
        """Compute perceptual hash for deduplication."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Compute pHash
                phash = imagehash.phash(img)
                return str(phash)
        except Exception as e:
            logger.warning(f"Could not compute hash for {image_path}: {e}")
            return None

    def _load_dataset_records(self, dataset_name: str) -> pd.DataFrame:
        """Load successful download records for a dataset."""
        index_file = self.metadata_dir / f"{dataset_name}_index.csv"
        
        if not index_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(index_file)
        # Filter for successful downloads only
        successful = df[df['failure_reason'].isna()].copy()
        
        # Add dataset label
        successful['dataset'] = dataset_name
        
        return successful

    def _deduplicate_across_datasets(
        self, 
        all_records: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Remove duplicates across datasets using SHA256 and perceptual hashes."""
        console.print("[yellow]Deduplicating across datasets...[/yellow]")
        
        seen_sha256: Set[str] = set()
        seen_phash: Set[str] = set()
        deduplicated_records = []
        duplicate_counts = {}
        
        for dataset in all_records['dataset'].unique():
            duplicate_counts[dataset] = 0
        
        # Process records in random order to avoid bias towards any dataset
        shuffled_records = all_records.sample(frac=1, random_state=self.seed)
        
        for _, record in track(
            shuffled_records.iterrows(), 
            description="Deduplicating",
            total=len(shuffled_records)
        ):
            is_duplicate = False
            
            # Check SHA256 hash
            if pd.notna(record['sha256']) and record['sha256'] in seen_sha256:
                is_duplicate = True
            
            # Check perceptual hash if we have the image file
            if not is_duplicate and pd.notna(record['final_path']):
                image_path = self.base_dir / record['final_path']
                if image_path.exists():
                    phash = self._compute_perceptual_hash(image_path)
                    if phash and phash in seen_phash:
                        is_duplicate = True
                    elif phash:
                        seen_phash.add(phash)
            
            if is_duplicate:
                duplicate_counts[record['dataset']] += 1
            else:
                deduplicated_records.append(record)
                if pd.notna(record['sha256']):
                    seen_sha256.add(record['sha256'])
        
        deduplicated_df = pd.DataFrame(deduplicated_records)
        
        console.print(f"Removed {len(all_records) - len(deduplicated_df)} duplicates")
        for dataset, count in duplicate_counts.items():
            if count > 0:
                console.print(f"  {dataset}: {count} duplicates")
        
        return deduplicated_df, duplicate_counts

    def _sample_dataset(self, records: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Sample N images from a dataset."""
        if len(records) <= self.n_per_dataset:
            console.print(f"Using all {len(records)} images from {dataset_name}")
            return records
        
        # Random sample
        sampled = records.sample(n=self.n_per_dataset, random_state=self.seed)
        console.print(f"Sampled {len(sampled)} images from {dataset_name}")
        return sampled

    def _create_train_val_splits(self, sampled_records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/val splits for each dataset."""
        train_records = []
        val_records = []
        
        for dataset in sampled_records['dataset'].unique():
            dataset_records = sampled_records[sampled_records['dataset'] == dataset]
            
            # Ensure we have enough records for validation
            actual_val_size = min(self.val_size, len(dataset_records) // 5)  # Max 20% for val
            
            if len(dataset_records) <= actual_val_size:
                # Too few records, use all for training
                train_records.append(dataset_records)
                console.print(f"⚠️  {dataset}: Too few images, using all {len(dataset_records)} for training")
            else:
                # Split into train/val
                train, val = train_test_split(
                    dataset_records,
                    test_size=actual_val_size,
                    random_state=self.seed,
                    stratify=None
                )
                
                train_records.append(train)
                val_records.append(val)
                console.print(f"✅ {dataset}: {len(train)} train, {len(val)} val")
        
        train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
        val_df = pd.concat(val_records, ignore_index=True) if val_records else pd.DataFrame()
        
        return train_df, val_df

    def _save_manifests(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Save train/val manifests."""
        # Create manifests with required columns
        train_manifest = train_df[['dataset', 'final_path', 'sha256', 'width', 'height']].copy()
        train_manifest['split'] = 'train'
        
        val_manifest = val_df[['dataset', 'final_path', 'sha256', 'width', 'height']].copy()
        val_manifest['split'] = 'val'
        
        # Save manifests
        train_file = self.sampled_dir / "train_manifest.csv"
        val_file = self.sampled_dir / "val_manifest.csv"
        combined_file = self.sampled_dir / "combined_manifest.csv"
        
        train_manifest.to_csv(train_file, index=False)
        val_manifest.to_csv(val_file, index=False)
        
        # Combined manifest
        combined_manifest = pd.concat([train_manifest, val_manifest], ignore_index=True)
        combined_manifest.to_csv(combined_file, index=False)
        
        console.print(f"✅ Saved manifests:")
        console.print(f"  Train: {train_file} ({len(train_manifest)} images)")
        console.print(f"  Val: {val_file} ({len(val_manifest)} images)")
        console.print(f"  Combined: {combined_file} ({len(combined_manifest)} images)")

    def run_sampling(self) -> Dict:
        """Run the complete sampling pipeline."""
        console.print(f"\n[bold green]Starting dataset sampling[/bold green]")
        console.print(f"Target: {self.n_per_dataset} images per dataset")
        console.print(f"Validation: {self.val_size} images per dataset")
        console.print(f"Available datasets: {', '.join(self.available_datasets)}")
        
        if not self.available_datasets:
            console.print("[red]No datasets available! Please run downloads first.[/red]")
            return {'status': 'failed', 'reason': 'No datasets available'}
        
        # Load all dataset records
        all_records = []
        for dataset_name in self.available_datasets:
            records = self._load_dataset_records(dataset_name)
            if not records.empty:
                all_records.append(records)
        
        if not all_records:
            console.print("[red]No successful downloads found![/red]")
            return {'status': 'failed', 'reason': 'No successful downloads'}
        
        combined_records = pd.concat(all_records, ignore_index=True)
        console.print(f"Loaded {len(combined_records)} total records")
        
        # Deduplicate across datasets
        deduplicated_records, duplicate_counts = self._deduplicate_across_datasets(combined_records)
        
        # Sample from each dataset
        sampled_records = []
        for dataset_name in self.available_datasets:
            dataset_records = deduplicated_records[deduplicated_records['dataset'] == dataset_name]
            if not dataset_records.empty:
                sampled = self._sample_dataset(dataset_records, dataset_name)
                sampled_records.append(sampled)
        
        if not sampled_records:
            console.print("[red]No records after sampling![/red]")
            return {'status': 'failed', 'reason': 'No records after sampling'}
        
        final_sampled = pd.concat(sampled_records, ignore_index=True)
        
        # Create train/val splits
        train_df, val_df = self._create_train_val_splits(final_sampled)
        
        # Save manifests
        self._save_manifests(train_df, val_df)
        
        # Summary
        summary = {
            'status': 'success',
            'datasets': self.available_datasets,
            'total_records': len(combined_records),
            'after_dedup': len(deduplicated_records),
            'duplicates_removed': duplicate_counts,
            'final_sampled': len(final_sampled),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'train_manifest': str(self.sampled_dir / "train_manifest.csv"),
            'val_manifest': str(self.sampled_dir / "val_manifest.csv"),
        }
        
        console.print(f"\n[bold green]Sampling Complete![/bold green]")
        console.print(f"Final dataset sizes:")
        for dataset in self.available_datasets:
            train_count = len(train_df[train_df['dataset'] == dataset])
            val_count = len(val_df[val_df['dataset'] == dataset])
            console.print(f"  {dataset}: {train_count} train, {val_count} val")
        
        return summary


def main():
    """Main function for dataset sampler."""
    parser = argparse.ArgumentParser(description="Sample and preprocess datasets")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="~/dataset_bias_data",
        help="Base directory for data"
    )
    parser.add_argument(
        "--n-per-dataset",
        type=int,
        default=10000,
        help="Number of images per dataset"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Validation set size per dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sampler and run
    sampler = DatasetSampler(
        base_dir=args.base_dir,
        n_per_dataset=args.n_per_dataset,
        val_size=args.val_size,
        seed=args.seed
    )
    
    results = sampler.run_sampling()
    
    if results['status'] == 'success':
        console.print("[bold green]Sampling completed successfully![/bold green]")
    else:
        console.print(f"[bold red]Sampling failed: {results['reason']}[/bold red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
