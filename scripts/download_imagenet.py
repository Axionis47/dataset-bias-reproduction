#!/usr/bin/env python3
"""
ImageNet-1K Dataset Downloader
Handles ImageNet download (requires registration and agreement).
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main():
    """Main function for ImageNet downloader."""
    parser = argparse.ArgumentParser(description="Download ImageNet-1K dataset")
    parser.add_argument(
        "--credentials-file",
        type=str,
        help="Path to file containing ImageNet credentials"
    )
    
    args = parser.parse_args()

    console.print("[bold red]ImageNet-1K Download[/bold red]")
    console.print("\nImageNet-1K requires registration and agreement to terms of use.")
    console.print("Please visit: https://www.image-net.org/download.php")
    console.print("\nSteps:")
    console.print("1. Register for an account")
    console.print("2. Agree to the terms of use")
    console.print("3. Download the dataset manually")
    console.print("4. Place images in: ~/dataset_bias_data/raw/imagenet/images/")
    
    console.print(f"\n[yellow]SKIPPING ImageNet download - requires manual setup[/yellow]")
    console.print("The reproduction will continue with available datasets.")
    
    # Create marker file to indicate this dataset was skipped
    marker_file = Path("~/dataset_bias_data/metadata/imagenet_skipped.txt").expanduser()
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_text("ImageNet-1K skipped - requires manual registration and download")
    
    return {
        'dataset': 'imagenet',
        'status': 'SKIPPED',
        'reason': 'Requires manual registration and download',
        'instructions': 'Visit https://www.image-net.org/download.php'
    }


if __name__ == "__main__":
    main()
