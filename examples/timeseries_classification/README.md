# Timeseries Classification for Wheelbot Dataset

This directory contains examples for training transformer models for timeseries classification on IMU sensor data.

## Classification Tasks

Three classification tasks are available:

1. **Surface Classification** (`--task surface`)
   - Classifies surface type: black_pvc, concrete, gray_felt
   - Uses yaw-related groups: yaw, yaw_circle, yaw_figure_eight
   - Balanced dataset (~3k samples per class)

2. **Robot Classification** (`--task robot`)
   - Classifies robot ID: wheelbot-beta-1, wheelbot-beta-2, wheelbot-beta-3
   - Only uses black_pvc surface data
   - Uses yaw-related groups: yaw, yaw_circle, yaw_figure_eight
   - Balanced dataset

3. **Group Classification** (`--task group`)
   - Classifies control mode: autonomous vs human
   - Autonomous: yaw, yaw_circle, yaw_figure_eight
   - Human: yaw_human
   - Only uses black_pvc surface data
   - Balanced dataset

## Usage

### 1. Export Datasets

First, export all three classification datasets:

```bash
python export_dataset.py
```

This will create three pickle files in the `dataset/` directory:
- `surface_classification_dataset.pkl`
- `robot_classification_dataset.pkl`
- `group_classification_dataset.pkl`

### 2. Train Models

Train models for a specific task with different sequence lengths:

```bash
# Surface classification
python train_transformer.py --task surface

# Robot classification
python train_transformer.py --task robot

# Group classification
python train_transformer.py --task group
```

Each training run will:
- Train 5 models with sequence lengths: 2, 5, 10, 20, 50
- Save models to `models/{task}_classifier_seq{N}.eqx`
- Generate individual training curves for each sequence length
- Create a comparison plot: accuracy vs sequence length
- Save results to `results/{task}_sequence_length_comparison.pkl`

## Model Architecture

Tiny Transformer:
- Input projection: 6 features → 64 dims (1 IMU sensor with 3-axis gyro + 3-axis accel)
- Learnable positional encoding
- 2 transformer encoder blocks:
  - 4-head multi-head attention
  - Layer normalization
  - Feedforward network (64 → 128 → 64) with GELU
  - Dropout (0.1)
- Global average pooling
- Linear classifier head

## Training Parameters

- Sequence lengths evaluated: 2, 5, 10, 20, 50 timesteps
- Batch size: 1024
- Learning rate: 1e-3 (with cosine decay)
- Optimizer: AdamW
- Epochs: 100
- Dropout: 0.1

## Output Files

### Models
- `models/{task}_classifier_seq{N}.eqx` - Trained model weights
- `models/{task}_classifier_seq{N}_hyperparams.json` - Hyperparameters

### Plots
- `plots/{task}_training_seq{N}.pdf` - Individual training curves
- `plots/{task}_accuracy_vs_sequence_length.pdf` - Comparison plot

### Results
- `results/{task}_sequence_length_comparison.pkl` - All training data for analysis
