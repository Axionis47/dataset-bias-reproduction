#!/usr/bin/env python3
"""
Smoke test for dataset bias reproduction.
Quick end-to-end test with minimal data.
"""

import sys
import tempfile
import time
from pathlib import Path

import torch
from rich.console import Console

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_factory import ModelFactory

console = Console()


def test_mps_availability():
    """Test MPS availability and basic operations."""
    console.print("[bold blue]Testing MPS Availability[/bold blue]")
    
    if torch.backends.mps.is_available():
        console.print("✅ MPS is available")
        
        # Test basic tensor operations
        device = torch.device('mps')
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.mm(x, y)
        
        console.print(f"✅ MPS tensor operations work: {z.shape}")
        return True
    else:
        console.print("❌ MPS not available, will use CPU")
        return False


def test_model_creation():
    """Test model creation and basic forward pass."""
    console.print("\n[bold blue]Testing Model Creation[/bold blue]")
    
    # Test all supported models
    models_to_test = ['convnext_tiny', 'resnet50', 'alexnet']
    
    for model_name in models_to_test:
        try:
            model = ModelFactory.create_model(
                model_name=model_name,
                num_classes=3,
                pretrained=False
            )
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 3), f"Expected (2, 3), got {output.shape}"
            console.print(f"✅ {model_name}: {output.shape}")
            
        except Exception as e:
            console.print(f"❌ {model_name}: {e}")
            return False
    
    return True


def test_transforms():
    """Test data transforms."""
    console.print("\n[bold blue]Testing Data Transforms[/bold blue]")
    
    try:
        from torchvision import transforms
        from PIL import Image
        
        # Create dummy image
        dummy_img = Image.new('RGB', (256, 256), color='red')
        
        # Test training transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Test validation transforms
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_tensor = train_transform(dummy_img)
        val_tensor = val_transform(dummy_img)
        
        assert train_tensor.shape == (3, 224, 224), f"Train transform shape: {train_tensor.shape}"
        assert val_tensor.shape == (3, 224, 224), f"Val transform shape: {val_tensor.shape}"
        
        console.print("✅ Data transforms work correctly")
        return True
        
    except Exception as e:
        console.print(f"❌ Transform test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    console.print("\n[bold blue]Testing Configuration Loading[/bold blue]")
    
    try:
        import yaml
        
        config_path = Path("experiments/Combo-1.yaml")
        if not config_path.exists():
            console.print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['experiment', 'data', 'model', 'training', 'hardware']
        for section in required_sections:
            if section not in config:
                console.print(f"❌ Missing config section: {section}")
                return False
        
        console.print("✅ Configuration loading works")
        return True
        
    except Exception as e:
        console.print(f"❌ Config loading failed: {e}")
        return False


def test_directory_structure():
    """Test directory structure creation."""
    console.print("\n[bold blue]Testing Directory Structure[/bold blue]")
    
    try:
        base_dir = Path("~/dataset_bias_data").expanduser()
        
        # Expected directories
        expected_dirs = [
            base_dir / "metadata",
            base_dir / "raw",
            base_dir / "sampled",
        ]
        
        for dir_path in expected_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            if not dir_path.exists():
                console.print(f"❌ Could not create directory: {dir_path}")
                return False
        
        console.print("✅ Directory structure creation works")
        return True
        
    except Exception as e:
        console.print(f"❌ Directory structure test failed: {e}")
        return False


def main():
    """Run smoke test."""
    console.print("[bold green]🚀 Dataset Bias Reproduction - Smoke Test[/bold green]")
    console.print("Running quick end-to-end verification...\n")
    
    start_time = time.time()
    
    tests = [
        ("MPS Availability", test_mps_availability),
        ("Model Creation", test_model_creation),
        ("Data Transforms", test_transforms),
        ("Configuration Loading", test_config_loading),
        ("Directory Structure", test_directory_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                console.print(f"[red]❌ {test_name} failed[/red]")
        except Exception as e:
            console.print(f"[red]❌ {test_name} failed with exception: {e}[/red]")
    
    elapsed = time.time() - start_time
    
    console.print(f"\n[bold]Smoke Test Results[/bold]")
    console.print(f"Passed: {passed}/{total}")
    console.print(f"Time: {elapsed:.2f} seconds")
    
    if passed == total:
        console.print("[bold green]✅ All smoke tests passed![/bold green]")
        console.print("The system is ready for dataset bias reproduction.")
        return 0
    else:
        console.print(f"[bold red]❌ {total - passed} tests failed[/bold red]")
        console.print("Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
