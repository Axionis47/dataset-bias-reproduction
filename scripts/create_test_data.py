#!/usr/bin/env python3
"""
Create minimal test data for pipeline verification.
Generates synthetic images for testing the complete pipeline.
"""

import hashlib
import random
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rich.console import Console

console = Console()


def create_synthetic_image(width: int = 224, height: int = 224, dataset_name: str = "test") -> Image.Image:
    """Create a synthetic image with dataset-specific characteristics."""
    # Create base image
    if dataset_name == "yfcc":
        # Flickr-style: more colorful, varied compositions
        base_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    elif dataset_name == "cc":
        # Web content: more neutral colors
        base_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    elif dataset_name == "datacomp":
        # Common crawl: mixed characteristics
        base_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        base_color = (128, 128, 128)
    
    img = Image.new('RGB', (width, height), base_color)
    draw = ImageDraw.Draw(img)
    
    # Add some patterns to make datasets distinguishable
    if dataset_name == "yfcc":
        # Add circles (photography-like)
        for _ in range(random.randint(2, 5)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(10, 50)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    elif dataset_name == "cc":
        # Add rectangles (web graphics-like)
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, width//2)
            y1 = random.randint(0, height//2)
            x2 = random.randint(width//2, width)
            y2 = random.randint(height//2, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    elif dataset_name == "datacomp":
        # Add lines (diverse content)
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))
    
    return img


def create_test_dataset(dataset_name: str, num_images: int = 50, base_dir: str = "~/dataset_bias_data"):
    """Create a test dataset with synthetic images."""
    base_path = Path(base_dir).expanduser()
    
    # Create directories
    raw_dir = base_path / "raw" / dataset_name / "images"
    metadata_dir = base_path / "metadata"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Creating {num_images} synthetic images for {dataset_name}...")
    
    records = []
    
    for i in range(num_images):
        # Create synthetic image
        img = create_synthetic_image(dataset_name=dataset_name)
        
        # Save image
        image_id = f"{dataset_name}_{i:06d}"
        image_path = raw_dir / f"{image_id}.jpg"
        img.save(image_path, "JPEG", quality=85)
        
        # Calculate hash
        with open(image_path, 'rb') as f:
            content = f.read()
            sha256 = hashlib.sha256(content).hexdigest()
        
        # Create record
        record = {
            'url': f'https://example.com/{dataset_name}/{image_id}.jpg',
            'image_id': image_id,
            'http_status': 200,
            'final_path': str(image_path.relative_to(base_path)),
            'bytes': len(content),
            'sha256': sha256,
            'width': img.width,
            'height': img.height,
            'failure_reason': None,
            'timestamp': time.time(),
        }
        
        records.append(record)
    
    # Save index
    df = pd.DataFrame(records)
    index_file = metadata_dir / f"{dataset_name}_index.csv"
    df.to_csv(index_file, index=False)
    
    console.print(f"✅ Created {num_images} images for {dataset_name}")
    console.print(f"   Images: {raw_dir}")
    console.print(f"   Index: {index_file}")
    
    return len(records)


def main():
    """Create test datasets."""
    console.print("[bold green]Creating Test Data for Pipeline Verification[/bold green]")
    
    # Create test datasets
    datasets = [
        ("yfcc", 30),
        ("cc", 30), 
        ("datacomp", 30),
    ]
    
    total_images = 0
    
    for dataset_name, num_images in datasets:
        count = create_test_dataset(dataset_name, num_images)
        total_images += count
    
    console.print(f"\n[bold green]Test Data Creation Complete![/bold green]")
    console.print(f"Total images created: {total_images}")
    console.print(f"Datasets: {[d[0] for d in datasets]}")
    console.print("\nYou can now run the pipeline with:")
    console.print("  make sample")
    console.print("  make train combo=Combo-1")
    console.print("  make eval combo=Combo-1")


if __name__ == "__main__":
    main()
