# Dynamics Model Learning

This directory contains code for learning neural network dynamics models of the wheelbot using JAX and Equinox. The models predict how the robot's state evolves given the current state and action.

## Overview

The dynamics model learns the function:
```
Δstate = f(state, action)
next_state = state + Δstate
```

Where:
- **state**: [yaw, roll, pitch, yaw_vel, roll_vel, pitch_vel, drive_wheel, reaction_wheel, drive_wheel_vel, reaction_wheel_vel] (10D)
- **action**: [drive_wheel_torque, reaction_wheel_torque] (2D)
- **Δstate**: Predicted change in state (10D)

Two training approaches are implemented:
1. **One-step prediction**: Minimize error on single-step predictions
2. **Multi-step rollout**: Minimize error over extended trajectory rollouts with curriculum learning

## Files

### Core Implementation

- **`modellearning_common.py`** - Shared utilities for all model learning tasks
  - `DynamicsDataset`: Dataset wrapper with batching support
  - `create_mlp_model()`: Create MLP dynamics model
  - `rollout_model()`: Roll out model over action sequences
  - `save_dynamics_model()` / `load_dynamics_model()`: Model serialization
  - `compute_normalization_params()`: Calculate normalization statistics
  - `normalize_data()`: Normalize states and actions

### Dataset Export

- **`export_dataset.py`** - Export training data from raw experiments
  - Processes experiments from `data/` (train) and `data_test/` (test)
  - Applies filtering, resampling, and time cutting
  - Exports two datasets:
    - `dataset/dataset_1_step.pkl`: One-step prediction dataset
    - `dataset/multistep_rollout_dataset.pkl`: 100-step rollout dataset
  
### Training Scripts

- **`modellearning_multistep.py`** - Multi-step rollout training with curriculum learning
  - Trains model with increasing rollout lengths (curriculum)
  - Linearly increases from 1-step to max_length in first half of training
  - Trains on full rollout length in second half
  - Memory-efficient batching with loss masking (avoids JIT recompilation)
  - Saves model to `models/trained_model_multistep.eqx`

### Evaluation

- **`modellearning_eval.py`** - Comprehensive model evaluation and plotting
  - `evaluate_rollout()`: Compute rollout MSE
  - `plot_state_comparison()`: Plot predicted vs. actual states over time
  - `plot_error_comparison()`: Compare one-step vs. multi-step model errors
  - `create_paper_plots()`: Generate publication-quality comparison plots
  - Generates plots in `plots/` directory

## Usage

### 1. Export Dataset

First, export the training and test datasets from raw experiment data:

```bash
python export_dataset.py
```

This creates:
- `dataset/dataset_1_step.pkl` - For one-step prediction training (~1-2GB)
- `dataset/multistep_rollout_dataset.pkl` - For rollout training (~1-2GB)

**Prerequisites**: Requires `data/` and `data_test/` directories with experiment data.

### 2. Train Multi-Step Model

Train a dynamics model using curriculum-based multi-step rollout loss:

```bash
python modellearning_multistep.py
```

**Configuration** (modify in `__main__` block):
```python
MAX_ROLLOUT_LENGTH = 50      # Maximum rollout length
MODEL_WIDTH_SIZE = 128        # MLP hidden layer width
MODEL_DEPTH = 2               # MLP depth
NUM_EPOCHS = 100              # Training epochs
BATCH_SIZE = 256              # Batch size
LEARNING_RATE = 1e-3          # Initial learning rate
```

**Output**:
- `models/trained_model_multistep.eqx` - Trained model
- `plots/multistep_training_loss.pdf` - Training curves
- `plots/multistep_curriculum_schedule.pdf` - Curriculum progression

### 3. Evaluate Model

Evaluate trained models and generate comparison plots:

```bash
python modellearning_eval.py
```

**Output plots**:
- `plots/paper_plot_comparison_first_3.pdf` - First 3 test sequences
- `plots/paper_plot_comparison_last_3.pdf` - Last 3 test sequences  
- `plots/paper_plot_error_histogram.pdf` - Error distribution comparison

## Dataset Format

Both dataset files contain pickled dictionaries with the structure:

```python
{
    "train": {
        "states": np.ndarray,      # (N, nx) or (N, nx) initial states
        "actions": np.ndarray,     # (N, nu) or (N, T, nu) action sequences
        "nextstates": np.ndarray,  # (N, nx) or (N, T, nx) target states
    },
    "test": {
        "states": np.ndarray,
        "actions": np.ndarray,
        "nextstates": np.ndarray,
    },
    "states_labels": List[str],    # State field names
    "actions_labels": List[str],   # Action field names
    "dt": float                     # Timestep (0.01s)
}
```

**One-step dataset** (`dataset_1_step.pkl`):
- `states`: (N, 10) - Initial states
- `actions`: (N, 2) - Single actions
- `nextstates`: (N, 10) - Next states

**Multi-step dataset** (`multistep_rollout_dataset.pkl`):
- `states`: (N, 10) - Initial states
- `actions`: (N, 100, 2) - 100-step action sequences
- `nextstates`: (N, 100, 10) - 100-step state trajectories

## Model Architecture

**MLP Structure** (default):
- Input: state + action = 12D
- Hidden layers: 128 units × 2 layers
- Output: state delta = 10D
- Activation: ReLU (Equinox default)

**Normalization**: All inputs and outputs are normalized to zero mean and unit variance during training.

## Training Details

### Multi-Step Curriculum Learning

The curriculum schedule linearly increases rollout length:

```
Epoch 0-50:   Rollout length increases from 1 → 50 steps
Epoch 50-100: Rollout length fixed at 50 steps
```

**Benefits**:
- Stabilizes early training (easier one-step task first)
- Gradually introduces long-term prediction challenges
- Better final performance on long rollouts

**Loss masking**: Rolls out full sequence every iteration but masks loss for steps beyond curriculum length. This avoids JIT recompilation overhead.

### Optimizer

- **Algorithm**: AdamW with cosine decay
- **Learning rate**: 1e-3 initial, decays to 0
- **Batch size**: 256 (multi-step) or 1024 (one-step)
- **Epochs**: 100

## Performance Metrics

Models are evaluated using:
1. **Mean Squared Error (MSE)**: Average squared error over rollout
2. **Per-timestep error**: Error at each step in the rollout
3. **Visual trajectory comparison**: Predicted vs. actual states

**Expected Results**:
- Multi-step models typically outperform one-step models on long rollouts
- One-step models may have lower initial error but accumulate drift
- Roll and pitch predictions are typically more accurate than yaw (no drift correction)

## Output Files

### Models
- `models/trained_model_multistep.eqx` - Multi-step trained model with hyperparameters

### Plots
- `plots/multistep_training_loss.pdf` - Training/test loss curves
- `plots/multistep_curriculum_schedule.pdf` - Curriculum rollout length over training
- `plots/paper_plot_comparison_first_3.pdf` - First 3 test rollout comparisons
- `plots/paper_plot_comparison_last_3.pdf` - Last 3 test rollout comparisons
- `plots/paper_plot_error_histogram.pdf` - Error distribution analysis

## Dependencies

- JAX (with GPU support recommended)
- Equinox (neural network library)
- Optax (optimization library)
- NumPy
- Matplotlib
- SciPy (for filtering)
- pickle (standard library)

## Notes

- **Memory**: Multi-step training requires ~8-16GB GPU memory for batch_size=256
- **Training time**: ~10-30 minutes on GPU, 1-2 hours on CPU
- **Data preprocessing**: Experiments are filtered by success status, resampled to 100Hz, and trimmed by 2s margins
- **Normalization**: Critical for training stability - all data is normalized to N(0,1)

## Future Improvements

Possible extensions:
- Add recurrent models (LSTM, GRU) for better long-term predictions
- Implement ensemble models for uncertainty quantification
- Add domain randomization for robustness
- Integrate model-based control or planning
