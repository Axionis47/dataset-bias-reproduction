# Dataset Bias Reproduction: A Decade's Battle on Dataset Bias

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-green.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive reproduction of **"A Decade's Battle on Dataset Bias: Are We There Yet?"** by Zhuang Liu and Kaiming He, optimised for laptop-scale experiments on Mac M4 with 16GB RAM.

## 🎯 Objective

This project reproduces the dataset classification experiments from the original paper, demonstrating that modern neural networks can achieve surprisingly high accuracy (>80%) in classifying which dataset an image comes from, even with large-scale, diverse datasets.

## 📊 Key Findings from Original Paper

- **Dataset Classification Accuracy:** 84.7% on YCD combination (YFCC + CC + DataComp)
- **Human Performance:** 45.4% average (vs 33.3% chance)
- **Architecture Agnostic:** Works across AlexNet, ResNet, ViT, ConvNeXt
- **Generalisation:** Models learn transferable patterns, not just memorisation
- **Robustness:** Survives various image corruptions

## 🗂️ Datasets Used

| Dataset | Description | Scale | Source |
|---------|-------------|-------|---------|
| **YFCC100M** | Flickr images | 100M images | Flickr uploads |
| **CC12M** | Conceptual Captions | 12M image-text pairs | Internet crawl |
| **DataComp-1B** | DataComp dataset | 1B image-text pairs | Common Crawl |
| **WIT** | Wikipedia Image Text | 11.5M image-text pairs | Wikipedia |
| **LAION-2B** | Large-scale AI Open Network | 2B image-text pairs | Common Crawl |
| **ImageNet-1K** | ImageNet classification | 14M images | Search engines |

## 🚀 Quick Start

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 16GB+ RAM recommended
- 100GB+ free storage

### Installation

```bash
# Clone the repository
git clone https://github.com/Axionis47/dataset-bias-reproduction.git
cd dataset-bias-reproduction

# Set up environment and dependencies
make setup

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Running Experiments

```bash
# Download datasets (interactive - will prompt for credentials if needed)
make download

# Sample and preprocess data
make sample

# Run Combo-1 experiment (YFCC + CC + DataComp)
make train combo=Combo-1

# Evaluate model
make eval combo=Combo-1

# Run corruption robustness evaluation
make corrupt_eval combo=Combo-1

# Generate comprehensive report
make report
```

### Smoke Test (3-minute end-to-end verification)
```bash
make smoke
```

## 📁 Project Structure

```
├── data/
│   ├── metadata/           # URL lists, indices, manifests
│   ├── raw/<dataset>/      # Downloaded originals
│   └── sampled/<dataset>/  # Sampled subsets (train/val)
├── experiments/            # YAML configs, checkpoints, outputs
├── notebooks/              # Analysis notebooks
├── reports/                # REPORT.md, figures, ethics notes
├── scripts/                # CLI helpers
├── src/
│   ├── data/              # Loaders, samplers, manifests
│   ├── models/            # Model zoo & factory
│   ├── train/             # Training loop, schedulers
│   └── eval/              # Clean eval + corruption eval
├── tests/                 # Pytest tests
└── Makefile              # Common commands
```

## 🔬 Experiments

### Primary Combinations
- **Combo-1:** YFCC + CC + DataComp (3-way classification)
- **Combo-2:** Add WIT (4-way classification)
- **All-6:** All available datasets (6-way classification)

### Ablation Studies
- **Data Size:** 1K, 5K, 10K images per dataset
- **Augmentations:** None → RandAug → RandAug + MixUp/CutMix
- **Architectures:** AlexNet, ResNet-50, ConvNeXt-Tiny

### Corruption Robustness Suite
Evaluates model robustness across 12 corruption types at 5 severity levels:
- **Noise:** Gaussian, shot, impulse
- **Blur:** Defocus, motion, zoom
- **Compression:** JPEG, pixelate
- **Photometric:** Brightness, contrast, saturation, hue

## 🛠️ Laptop Optimisations

- **Scaled sample sizes:** 10K images/dataset (vs 1M in paper)
- **Reduced epochs:** 30 (vs 300 in paper)
- **MPS acceleration:** Apple Silicon GPU support
- **Gradient accumulation:** Effective batch size with memory constraints
- **Efficient I/O:** 500px thumbnail caching

## 📈 Expected Results

Based on paper scaling, we expect:
- **Combo-1 (YCD):** ~75-80% accuracy (vs 84.7% in paper)
- **Human baseline:** ~45% accuracy
- **Corruption robustness:** Graceful degradation with severity
- **Transfer learning:** Non-trivial ImageNet-1K linear probe accuracy

## 🔍 Key Features

- ✅ **Resumable downloads** with validation and retry logic
- ✅ **Cross-dataset deduplication** using SHA256 + perceptual hashing
- ✅ **MPS-optimised training** with automatic fallback to CPU
- ✅ **Comprehensive logging** with Rich progress bars
- ✅ **Deterministic experiments** with seed management
- ✅ **Indian English reporting** with British spellings
- ✅ **CI/CD pipeline** with GitHub Actions
- ✅ **Ethics compliance** with dataset licence tracking

## 📊 Monitoring & Logging

- **Terminal:** Rich progress bars and status tables
- **Files:** Detailed logs in `logs/` directory
- **Metrics:** JSON outputs for programmatic analysis
- **Visualisation:** Matplotlib plots and confusion matrices

## 🧪 Testing

```bash
# Run all tests
pytest

# Run smoke test (fast end-to-end verification)
make smoke

# Run specific test categories
pytest tests/test_downloaders.py
pytest tests/test_samplers.py
```

## 📝 Citation

If you use this reproduction in your research, please cite both the original paper and this reproduction:

```bibtex
@article{liu2024decade,
  title={A Decade's Battle on Dataset Bias: Are We There Yet?},
  author={Liu, Zhuang and He, Kaiming},
  journal={arXiv preprint arXiv:2403.08632},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Original paper authors: Zhuang Liu and Kaiming He
- Dataset providers: YFCC, CC, DataComp, WIT, LAION, ImageNet teams
- PyTorch and timm library maintainers

---

**Repository:** https://github.com/Axionis47/dataset-bias-reproduction  
**Paper:** https://arxiv.org/abs/2403.08632  
**Original Code:** https://github.com/liuzhuang13/bias
