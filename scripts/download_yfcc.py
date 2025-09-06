#!/usr/bin/env python3
"""
YFCC100M Dataset Downloader
Downloads images from Yahoo Flickr Creative Commons 100M dataset.
"""

import argparse
import gzip
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


class YFCCDownloader(BaseDownloader):
    """Downloader for YFCC100M dataset."""

    def __init__(self, base_dir: str = "~/dataset_bias_data", **kwargs):
        super().__init__("yfcc", base_dir, **kwargs)
        # Official YFCC100M metadata URLs
        self.metadata_urls = [
            "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/yfcc100m_dataset-0.bz2",
            "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/yfcc100m_dataset-1.bz2",
            "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/yfcc100m_dataset-2.bz2",
            "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/yfcc100m_dataset-3.bz2",
            "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/yfcc100m_dataset-4.bz2",
        ]

    def get_dataset_info(self) -> Dict:
        """Get YFCC100M dataset information."""
        return {
            'name': 'YFCC100M',
            'full_name': 'Yahoo Flickr Creative Commons 100M',
            'description': '100 million Flickr images with Creative Commons licences',
            'source': 'Flickr user uploads',
            'licence': 'CC BY 2.0 (various Creative Commons licences)',
            'paper': 'Thomee et al. 2016 - YFCC100M: The new data in multimedia research',
            'url': 'https://multimediacommons.wordpress.com/',
            'size': '100M images',
            'ethical_status': 'APPROVED - Public Creative Commons content',
            'access_requirements': 'None - publicly available',
        }

    def _download_metadata_file(self, url: str, max_records: int = None) -> pd.DataFrame:
        """Download and parse a YFCC100M metadata file."""
        console.print(f"Downloading metadata from {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # YFCC100M format: tab-separated values
        # Fields: photo_id, user_nsid, user_nickname, date_taken, date_uploaded,
        #         capture_device, title, description, user_tags, machine_tags,
        #         longitude, latitude, accuracy, page_url, download_url, license_name,
        #         license_url, server_id, farm_id, secret, secret_original, extension,
        #         marker

        records = []

        if url.endswith('.bz2'):
            import bz2
            content = bz2.decompress(response.content).decode('utf-8')
        elif url.endswith('.gz'):
            content = gzip.decompress(response.content).decode('utf-8')
        else:
            content = response.text

        lines = content.strip().split('\n')

        for i, line in enumerate(lines):
            if max_records and i >= max_records:
                break

            fields = line.split('\t')
            if len(fields) >= 15:  # Minimum required fields
                try:
                    photo_id = fields[0]
                    download_url = fields[14] if len(fields) > 14 else None

                    # Only include records with valid download URLs
                    if download_url and download_url.startswith('http'):
                        records.append({
                            'photo_id': photo_id,
                            'url': download_url,
                            'title': fields[6] if len(fields) > 6 else '',
                            'tags': fields[8] if len(fields) > 8 else '',
                            'licence': fields[15] if len(fields) > 15 else '',
                        })
                except Exception as e:
                    logger.warning(f"Error parsing line {i}: {e}")
                    continue

        return pd.DataFrame(records)

    def get_url_list(self) -> List[Tuple[str, str]]:
        """Get list of YFCC100M URLs to download."""
        console.print("[yellow]Fetching YFCC100M metadata...[/yellow]")

        # Check if we have cached metadata
        metadata_file = self.metadata_dir / "yfcc_metadata.csv"

        if metadata_file.exists():
            console.print("Loading cached metadata...")
            df = pd.read_csv(metadata_file)
        else:
            console.print("Downloading YFCC100M metadata files...")

            # Download first metadata file (others are very large)
            # For laptop reproduction, we'll use a subset
            df = self._download_metadata_file(self.metadata_urls[0], max_records=50000)
            df.to_csv(metadata_file, index=False)
            console.print(f"Saved metadata to {metadata_file}")

        # Filter for valid URLs and convert to list
        df_valid = df[df['url'].notna() & df['url'].str.startswith('http')]

        urls_and_ids = [
            (row['url'], str(row['photo_id']))
            for _, row in df_valid.iterrows()
        ]

        console.print(f"Prepared {len(urls_and_ids)} YFCC URLs")
        return urls_and_ids


def main():
    """Main function for YFCC downloader."""
    parser = argparse.ArgumentParser(description="Download YFCC100M dataset")
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
    parser.add_argument(
        "--real-urls",
        action="store_true",
        help="Attempt to use real YFCC URLs (requires metadata)"
    )
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Create downloader
        downloader = YFCCDownloader(
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
        console.print(f"\n[bold green]YFCC Download Complete![/bold green]")
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
