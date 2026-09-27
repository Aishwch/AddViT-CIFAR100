# AddViT: CIFAR-100 Image Classification

This repository contains the implementation and evaluation of the Add-ViT model for image classification on the CIFAR-100 dataset.

## Project Overview

Add-ViT is a Vision Transformer based architecture that combines transformer-based attention with additional convolutional and attention mechanisms.

The project includes:
- Add-ViT model implementation
- CIFAR-100 training configuration
- Trained model evaluation
- Final evaluation notebook
- Single-image inference/demo

## Dataset

**Dataset:** CIFAR-100

- Training images: 50,000
- Test images: 10,000
- Number of classes: 100
- Image size: 32 × 32
- Image channels: 3

The CIFAR-100 dataset is not included in this repository.

## Model

The implementation is provided in:

`add_vit_model.py`

The training script is provided in:

`train_add_vit.py`

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset | CIFAR-100 |
| Model | Add-ViT Base |
| Epochs | 150 |
| Batch Size | 64 |
| Learning Rate | 0.0001 |
| Weight Decay | 0.05 |
| Warmup Epochs | 10 |
| Label Smoothing | 0.1 |
| Mixup Alpha | 1.0 |
| CutMix Alpha | 0.8 |
| Drop Path | 0.1 |
| Alpha PMSA | 0.5 |
| Seed | 42 |

## Results

The final model was evaluated on the CIFAR-100 test set containing 10,000 images.

```text
Correct predictions : 7107/10000
Test Accuracy       : 71.07%S