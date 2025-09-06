"""
Base downloader class for dataset bias reproduction.
Provides common functionality for all dataset downloaders.
"""

import asyncio
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tqdm import tqdm

console = Console()
logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """Base class for dataset downloaders with common functionality."""

    def __init__(
        self,
        dataset_name: str,
        base_dir: str = "~/dataset_bias_data",
        max_concurrent: int = 16,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        self.dataset_name = dataset_name
        self.base_dir = Path(base_dir).expanduser()
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Create directory structure
        self.raw_dir = self.base_dir / "raw" / dataset_name
        self.metadata_dir = self.base_dir / "metadata"
        self.images_dir = self.raw_dir / "images"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Index file for tracking downloads
        self.index_file = self.metadata_dir / f"{dataset_name}_index.csv"
        self.failed_file = self.metadata_dir / f"{dataset_name}_failed.csv"

        # Load existing index if available
        self.existing_index = self._load_existing_index()

    def _load_existing_index(self) -> pd.DataFrame:
        """Load existing download index for resumable downloads."""
        if self.index_file.exists():
            try:
                return pd.read_csv(self.index_file)
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _save_index(self, records: List[Dict]) -> None:
        """Save download index to CSV."""
        if records:
            df = pd.DataFrame(records)
            # Append to existing index
            if not self.existing_index.empty:
                df = pd.concat([self.existing_index, df], ignore_index=True)
            df.to_csv(self.index_file, index=False)

    def _save_failed(self, failed_records: List[Dict]) -> None:
        """Save failed downloads to separate CSV."""
        if failed_records:
            df = pd.DataFrame(failed_records)
            df.to_csv(self.failed_file, index=False)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _validate_image(self, file_path: Path) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Validate that file is a valid image and get dimensions."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                # Basic validation: non-zero size, reasonable dimensions
                if width > 0 and height > 0 and width < 50000 and height < 50000:
                    return True, (width, height)
                return False, None
        except Exception:
            return False, None

    def _get_file_extension(self, url: str, content_type: str = "") -> str:
        """Determine file extension from URL or content type."""
        # Try URL first
        parsed = urlparse(url)
        path = parsed.path.lower()
        if path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            return os.path.splitext(path)[1]

        # Try content type
        content_type = content_type.lower()
        if 'jpeg' in content_type or 'jpg' in content_type:
            return '.jpg'
        elif 'png' in content_type:
            return '.png'
        elif 'gif' in content_type:
            return '.gif'
        elif 'webp' in content_type:
            return '.webp'
        elif 'bmp' in content_type:
            return '.bmp'

        # Default to jpg
        return '.jpg'

    async def _download_single_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        image_id: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict:
        """Download a single image with validation and error handling."""
        async with semaphore:
            record = {
                'url': url,
                'image_id': image_id,
                'http_status': None,
                'final_path': None,
                'bytes': 0,
                'sha256': None,
                'width': None,
                'height': None,
                'failure_reason': None,
                'timestamp': time.time(),
            }

            # Check if already downloaded
            if not self.existing_index.empty:
                existing = self.existing_index[
                    (self.existing_index['url'] == url) |
                    (self.existing_index['image_id'] == image_id)
                ]
                if not existing.empty and existing.iloc[0]['failure_reason'] is None:
                    # Already successfully downloaded
                    return existing.iloc[0].to_dict()

            for attempt in range(self.retry_attempts):
                try:
                    async with session.get(url, timeout=30) as response:
                        record['http_status'] = response.status

                        if response.status != 200:
                            record['failure_reason'] = f"HTTP {response.status}"
                            continue

                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        if not any(t in content_type.lower() for t in ['image', 'jpeg', 'png', 'gif', 'webp']):
                            record['failure_reason'] = f"Invalid content type: {content_type}"
                            continue

                        # Read content
                        content = await response.read()
                        if len(content) == 0:
                            record['failure_reason'] = "Empty file"
                            continue

                        record['bytes'] = len(content)

                        # Determine file extension and path
                        ext = self._get_file_extension(url, content_type)
                        file_path = self.images_dir / f"{image_id}{ext}"

                        # Save file
                        with open(file_path, 'wb') as f:
                            f.write(content)

                        # Validate image
                        is_valid, dimensions = self._validate_image(file_path)
                        if not is_valid:
                            file_path.unlink()  # Delete invalid file
                            record['failure_reason'] = "Invalid image format"
                            continue

                        # Success!
                        record['final_path'] = str(file_path.relative_to(self.base_dir))
                        record['sha256'] = self._get_file_hash(file_path)
                        record['width'], record['height'] = dimensions
                        record['failure_reason'] = None
                        return record

                except asyncio.TimeoutError:
                    record['failure_reason'] = "Timeout"
                except Exception as e:
                    record['failure_reason'] = str(e)

                # Wait before retry
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

            return record

    async def download_batch(
        self,
        urls_and_ids: List[Tuple[str, str]],
        batch_name: str = "batch"
    ) -> Tuple[List[Dict], List[Dict]]:
        """Download a batch of images concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        successful_records = []
        failed_records = []

        # Create progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(
                f"[green]Downloading {self.dataset_name} {batch_name}",
                total=len(urls_and_ids)
            )

            # Create download tasks
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                connector=aiohttp.TCPConnector(limit=self.max_concurrent)
            ) as session:

                tasks = [
                    self._download_single_image(session, url, image_id, semaphore)
                    for url, image_id in urls_and_ids
                ]

                # Process completed downloads
                for coro in asyncio.as_completed(tasks):
                    record = await coro
                    progress.advance(task)

                    if record['failure_reason'] is None:
                        successful_records.append(record)
                    else:
                        failed_records.append(record)

        return successful_records, failed_records

    @abstractmethod
    def get_url_list(self) -> List[Tuple[str, str]]:
        """Get list of (URL, image_id) tuples to download.
        
        Must be implemented by each dataset downloader.
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict:
        """Get dataset information including licence, source, etc.
        
        Must be implemented by each dataset downloader.
        """
        pass

    def estimate_storage_requirements(self, num_images: int) -> Dict[str, float]:
        """Estimate storage requirements in GB."""
        # Conservative estimates based on typical image sizes
        avg_size_mb = {
            'yfcc': 0.5,      # Flickr images, typically compressed
            'cc': 0.3,        # Web images, smaller
            'datacomp': 0.4,  # Web crawl, mixed sizes
            'wit': 0.6,       # Wikipedia, higher quality
            'laion': 0.4,     # Web crawl, mixed
            'imagenet': 0.15, # Standardised, compressed
        }.get(self.dataset_name.lower(), 0.5)

        total_gb = (num_images * avg_size_mb) / 1024
        return {
            'estimated_total_gb': total_gb,
            'with_overhead_gb': total_gb * 1.2,  # 20% overhead
            'avg_size_mb': avg_size_mb,
        }

    def run_download(self, max_images: Optional[int] = None) -> Dict:
        """Run the complete download process."""
        console.print(f"\n[bold green]Starting {self.dataset_name} download[/bold green]")

        # Get dataset info
        dataset_info = self.get_dataset_info()
        console.print(f"Dataset: {dataset_info['name']}")
        console.print(f"Source: {dataset_info['source']}")
        console.print(f"Licence: {dataset_info['licence']}")

        # Get URL list
        console.print("\n[yellow]Fetching URL list...[/yellow]")
        urls_and_ids = self.get_url_list()

        if max_images:
            urls_and_ids = urls_and_ids[:max_images]

        console.print(f"Found {len(urls_and_ids)} URLs to download")

        # Estimate storage
        storage_info = self.estimate_storage_requirements(len(urls_and_ids))
        console.print(f"Estimated storage: {storage_info['estimated_total_gb']:.1f} GB")

        # Run download
        successful_records, failed_records = asyncio.run(
            self.download_batch(urls_and_ids)
        )

        # Save results
        self._save_index(successful_records)
        self._save_failed(failed_records)

        # Summary
        total = len(urls_and_ids)
        success_count = len(successful_records)
        failed_count = len(failed_records)

        console.print(f"\n[bold green]Download Summary for {self.dataset_name}[/bold green]")
        console.print(f"Total URLs: {total}")
        console.print(f"Successful: {success_count} ({success_count/total*100:.1f}%)")
        console.print(f"Failed: {failed_count} ({failed_count/total*100:.1f}%)")

        if successful_records:
            total_mb = sum(r['bytes'] for r in successful_records) / (1024 * 1024)
            console.print(f"Total downloaded: {total_mb:.1f} MB")

        return {
            'dataset': self.dataset_name,
            'total_urls': total,
            'successful': success_count,
            'failed': failed_count,
            'success_rate': success_count / total if total > 0 else 0,
            'total_mb': sum(r['bytes'] for r in successful_records) / (1024 * 1024),
            'index_file': str(self.index_file),
            'failed_file': str(self.failed_file),
        }
