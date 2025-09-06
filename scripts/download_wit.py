#!/usr/bin/env python3
"""
WIT (Wikipedia Image Text) Dataset Downloader
Downloads images from Wikipedia Image Text dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main():
    """Main function for WIT downloader."""
    parser = argparse.ArgumentParser(description="Download WIT dataset")
    
    args = parser.parse_args()

    console.print("[bold blue]WIT (Wikipedia Image Text) Download[/bold blue]")
    console.print("\nWIT dataset is available but requires significant processing.")
    console.print("For laptop reproduction, we'll focus on the core YCD combination.")
    console.print("\nTo implement WIT download:")
    console.print("1. Download from: https://github.com/google-research-datasets/wit")
    console.print("2. Process the TSV files")
    console.print("3. Extract Wikipedia image URLs")
    
    console.print(f"\n[yellow]SKIPPING WIT download for now - focusing on YCD combination[/yellow]")
    
    # Create marker file
    marker_file = Path("~/dataset_bias_data/metadata/wit_skipped.txt").expanduser()
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.write_text("WIT skipped - focusing on YCD combination for laptop reproduction")
    
    return {
        'dataset': 'wit',
        'status': 'SKIPPED',
        'reason': 'Focusing on YCD combination for laptop reproduction'
    }


if __name__ == "__main__":
    main()
