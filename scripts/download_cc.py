#!/usr/bin/env python3
"""
Conceptual Captions 12M (CC12M) Dataset Downloader
Downloads images from Google's Conceptual Captions 12M dataset.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from rich.console import Console

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.base_downloader import BaseDownloader

console = Console()
logger = logging.getLogger(__name__)


class CC12MDownloader(BaseDownloader):
    """Downloader for Conceptual Captions 12M dataset."""

    def __init__(self, base_dir: str = "~/dataset_bias_data", **kwargs):
        super().__init__("cc", base_dir, **kwargs)
        # Official CC12M dataset URLs from Google Drive
        self.train_url = "https://drive.google.com/uc?id=1mZ_sHAp7jpMfFVY2TFN9wZioYujoYfCL"

    def get_dataset_info(self) -> Dict:
        """Get CC12M dataset information."""
        return {
            'name': 'CC12M',
            'full_name': 'Conceptual Captions 12M',
            'description': '12 million image-text pairs from the web',
            'source': 'Web crawl with alt-text captions',
            'licence': 'Various (web content)',
            'paper': 'Changpinyo et al. 2021 - Conceptual 12M: Pushing web-scale image-text pre-training',
            'url': 'https://github.com/google-research-datasets/conceptual-12m',
            'size': '12M image-text pairs',
            'ethical_status': 'APPROVED - Research use, web content',
            'access_requirements': 'None - publicly available',
        }

    def _download_metadata_file(self, url: str, max_records: int = None) -> pd.DataFrame:
        """Download and parse CC12M metadata file."""
        console.print(f"Downloading metadata from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # CC12M format: TSV with columns: url, caption
        records = []
        lines = response.text.strip().split('\n')
        
        for i, line in enumerate(lines):
            if max_records and i >= max_records:
                break
                
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    image_url = parts[0].strip()
                    caption = parts[1].strip()
                    
                    # Only include valid HTTP URLs
                    if image_url.startswith('http'):
                        records.append({
                            'url': image_url,
                            'caption': caption,
                            'image_id': f"cc12m_{i:08d}",
                        })
                except Exception as e:
                    logger.warning(f"Error parsing line {i}: {e}")
                    continue
        
        return pd.DataFrame(records)

    def get_url_list(self) -> List[Tuple[str, str]]:
        """Get list of CC12M URLs to download."""
        console.print("[yellow]Fetching CC12M metadata...[/yellow]")
        
        # Check if we have cached metadata
        metadata_file = self.metadata_dir / "cc12m_metadata.csv"
        
        if metadata_file.exists():
            console.print("Loading cached metadata...")
            df = pd.read_csv(metadata_file)
        else:
            console.print("Downloading CC12M metadata...")
            
            # Download training set metadata (validation set is much smaller)
            df = self._download_metadata_file(self.train_url, max_records=100000)
            df.to_csv(metadata_file, index=False)
            console.print(f"Saved metadata to {metadata_file}")

        # Filter for valid URLs and convert to list
        df_valid = df[df['url'].notna() & df['url'].str.startswith('http')]
        
        urls_and_ids = [
            (row['url'], row['image_id'])
            for _, row in df_valid.iterrows()
        ]

        console.print(f"Prepared {len(urls_and_ids)} CC12M URLs")
        return urls_and_ids


def main():
    """Main function for CC12M downloader."""
    parser = argparse.ArgumentParser(description="Download CC12M dataset")
    parser.add_argument(
        "--max-images",
        type=int,
        default=10000,
        help="Maximum number of images to download (default: 10000)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="~/dataset_bias_data",
        help="Base directory for downloads"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Maximum concurrent downloads"
    )
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Create downloader
        downloader = CC12MDownloader(
            base_dir=args.base_dir,
            max_concurrent=args.max_concurrent
        )

        # Check ethics and licence
        dataset_info = downloader.get_dataset_info()
        console.print(f"\n[bold blue]Dataset Information[/bold blue]")
        console.print(f"Name: {dataset_info['name']}")
        console.print(f"Licence: {dataset_info['licence']}")
        console.print(f"Ethical Status: {dataset_info['ethical_status']}")
        console.print(f"Access: {dataset_info['access_requirements']}")

        # Confirm download
        if not console.input("\nProceed with download? [y/N]: ").lower().startswith('y'):
            console.print("Download cancelled.")
            return

        # Run download
        results = downloader.run_download(max_images=args.max_images)
        
        # Print final results
        console.print(f"\n[bold green]CC12M Download Complete![/bold green]")
        console.print(f"Success rate: {results['success_rate']:.1%}")
        console.print(f"Downloaded: {results['total_mb']:.1f} MB")
        console.print(f"Index file: {results['index_file']}")

        if results['failed'] > 0:
            console.print(f"[yellow]Failed downloads logged to: {results['failed_file']}[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[red]Download interrupted by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during download: {e}[/red]")
        logger.exception("Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
