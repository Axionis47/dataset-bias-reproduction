# Paper Analysis: "A Decade's Battle on Dataset Bias: Are We There Yet?"

**Authors:** Zhuang Liu, Kaiming He (Meta AI Research, FAIR)  
**Paper URL:** https://arxiv.org/abs/2403.08632  
**Code:** https://github.com/liuzhuang13/bias

## Executive Summary

This paper revisits the "dataset classification" experiment from Torralba and Efros (2011) using modern neural networks and large-scale datasets. The key finding is that neural networks can achieve surprisingly high accuracy (>80% in most cases) in classifying which dataset an image comes from, even with modern, diverse, and presumably less biased datasets.

## Datasets Used (Table 1)

The paper uses 6 large-scale datasets for the dataset classification task:

| Dataset | Description | Scale | Source |
|---------|-------------|-------|---------|
| **YFCC100M** [49] | 100M Flickr images | 100M images | Flickr user uploads |
| **CC12M** [4] | Conceptual Captions 12M | 12M image-text pairs | Internet crawl |
| **DataComp-1B** [15] | DataComp dataset | 1B image-text pairs | Common Crawl |
| **WIT** [45] | Wikipedia Image Text | 11.5M image-text pairs | Wikipedia |
| **LAION-2B** [42] | Large-scale AI Open Network | 2B image-text pairs | Common Crawl |
| **ImageNet-1K** [8] | ImageNet classification | 14M images | Search engines |

## Key Methodology

### Dataset Classification Task
- **Task Definition:** N-way classification where N = number of datasets
- **Default Setup:** 1M training images + 10K validation images per dataset
- **Evaluation:** Individual image classification (no batch statistics exploitation)

### Sampling Policy
- **Uniform sampling** per dataset to create balanced training/validation sets
- For datasets with pre-defined splits, sample only from training split
- Maintain same number of images across all datasets

### Preprocessing & Augmentation
- **Training:** 
  - Random resized crop to 224×224
  - RandAug (magnitude=9, probability=0.5)
  - MixUp (α=0.8) 
  - CutMix (α=1.0)
  - Label smoothing (0.1)
- **Validation:**
  - Resize shortest side to 256px (maintaining aspect ratio)
  - Centre crop to 224×224
  - No augmentations

### Training Recipe (Table 9)
- **Optimiser:** AdamW
- **Learning Rate:** 1e-3 (base)
- **Weight Decay:** 0.3
- **Momentum:** β₁=0.9, β₂=0.95
- **Batch Size:** 4096
- **Schedule:** Cosine decay with 20 epoch warmup
- **Epochs:** 300 (ImageNet-1K equivalent iterations)

### Model Architectures Tested
- **AlexNet:** 77.8% accuracy (YCD combination)
- **VGG-16:** 83.5% accuracy
- **ResNet-50:** 83.8% accuracy  
- **ViT-S:** 82.4% accuracy
- **ConvNeXt-Tiny:** 84.7% accuracy (default choice)

## Key Experimental Combinations

### Three-way Combinations (20 total)
All possible combinations of 3 datasets from the 6 available:
- **YCD (YFCC + CC + DataComp):** 84.7% accuracy (primary focus)
- **Best:** YFCC + CC + ImageNet: 92.7% accuracy
- **Worst:** CC + WIT + LAION: 62.8% accuracy
- **Range:** 62.8% - 92.7% (all significantly above 33.3% chance)

### Multi-way Combinations
- **4-way:** 79.1% accuracy
- **5-way:** 67.4% accuracy  
- **6-way (all datasets):** 69.2% accuracy

## Scaling Behaviours Observed

1. **More training data improves validation accuracy** (generalisation, not memorisation)
2. **Stronger data augmentation improves accuracy** (pattern learning, not overfitting)
3. **Larger models perform better** with diminishing returns
4. **Very small models** (7K parameters) still achieve 72.4% on YCD

## Robustness Analysis

### Corruption Types Tested
- **Colour jittering** (strength 1.0-2.0): 80.2-81.1% accuracy
- **Gaussian noise** (std 0.2-0.3): 75.1-77.3% accuracy  
- **Gaussian blur** (radius 3-5): 78.1-80.9% accuracy
- **Low resolution** (32×32, 64×64): 68.4-78.4% accuracy

Results suggest models learn beyond low-level signatures.

## Self-Supervised Learning Results

- **MAE pre-trained on ImageNet → Linear probe:** 76.2% accuracy
- **MAE pre-trained on YCD → Linear probe:** 78.4% accuracy
- **Fully supervised baseline:** 82.9% accuracy

## Transfer Learning to ImageNet-1K

Features learned from dataset classification show transferability:
- **Random weights:** 6.7% accuracy
- **YCD dataset classifier:** 27.7% accuracy
- **6-dataset classifier:** 34.8% accuracy
- **MAE (reference):** 68.0% accuracy
- **MoCo v3 (reference):** 76.7% accuracy

## Human Performance Baseline

- **User study:** 20 ML researchers (14 with CV experience)
- **Mean accuracy:** 45.4% (vs 33.3% chance, 84.7% neural network)
- **Range:** 40-50% for most participants
- **Task difficulty:** Rated as "difficult" by 15/20 participants

## Implementation Notes

- **Image preprocessing:** Resize shorter side to 500px for faster loading
- **Training iterations:** Same as 300-epoch ImageNet-1K training
- **Inference:** 256px shortest side resize → 224×224 centre crop
- **For image-text datasets:** Use only images, ignore text

## Key Insights for Reproduction

1. **Dataset bias is learnable** even with modern, diverse datasets
2. **Generalisation patterns exist** - not pure memorisation
3. **Architecture-agnostic** - works across different model families
4. **Scalable** - even small models show strong performance
5. **Robust** - survives various image corruptions
6. **Transferable** - learned features help with semantic tasks

## Laptop Scaling Considerations

- **Default 1M images/dataset** may need reduction to 10K-100K for storage/compute
- **Batch size 4096** needs reduction with gradient accumulation for 16GB RAM
- **300 epochs** can be scaled to 30-50 epochs
- **MPS backend** should be used for Apple Silicon acceleration
