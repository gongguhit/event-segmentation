# Lightweight Training Mode for Event Segmentation

This document explains how to use the lightweight training mode for the event segmentation model, which is designed to train much faster on limited hardware.

## Overview

The lightweight training mode includes several optimizations:

1. **Reduced dataset size**: Only uses the `outdoor_day1` sequence instead of all sequences
2. **Smaller model**: Uses a simplified model architecture with fewer parameters
3. **Downsampled images**: Optionally reduces image resolution to speed up processing
4. **Fewer epochs**: Trains for 10 epochs instead of 100
5. **Smaller batch size**: Uses a batch size of 4 instead of 8

These optimizations can reduce training time from 30+ hours per epoch to less than 1 hour per epoch on the same hardware.

## Usage

To use the lightweight training mode, run:

```bash
python main.py train --lite
```

This will automatically:
1. Use the `train_lite.py` script instead of `train.py`
2. Use the `config/lite.yaml` configuration instead of `config/default.yaml`
3. Use the `LiteEventSegmentationModel` instead of the full `EventSegmentationModel`

## Configuration

The lightweight configuration is defined in `config/lite.yaml`. You can modify this file to adjust:

- `data.batch_size`: Batch size for training (default: 4)
- `data.sequences`: Which sequences to use (default: ['outdoor_day1'])
- `data.downsample_factor`: Factor by which to downsample images (default: 2)
- `model.embedding_dims`: Embedding dimensions for each layer (default: [32, 64, 128, 256])
- `training.epochs`: Number of epochs to train (default: 10)

## Performance Comparison

| Configuration | Dataset Size | Model Parameters | Training Time per Epoch | GPU Memory Usage |
|---------------|--------------|------------------|-------------------------|------------------|
| Full          | 51,663 samples | ~5M             | ~30 hours              | ~4GB             |
| Lite          | 11,936 samples | ~1M             | ~1 hour                | ~1GB             |

## When to Use

Use the lightweight training mode when:

1. You want to quickly test changes to the model or training process
2. You have limited computational resources
3. You're debugging issues with the training pipeline
4. You want to perform rapid prototyping

For final model training and evaluation, you should use the full training mode.

## Limitations

The lightweight model has some limitations:

1. Lower accuracy due to the simplified architecture
2. Less generalization ability due to training on less data
3. Reduced resolution when using downsampling

However, it's still useful for development and testing purposes. 