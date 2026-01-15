# Wheelbot Dataset

A large, high-quality dynamics dataset of the [Mini Wheelbot](https://github.com/wheelbot/Mini-Wheelbot).
The dataset contains 1 kHz data of all onboard sensor readings, the estimated state, ground-truth pose measurements from a motion capture system, and a third-person view video of the experiment.
We perform a variety of experiments using pseudo-random binary excitation signals (PRBS) as setpoints of a linear controller, an MPC for driving, and an RL policy that races along a track.
Experiments are performed across multiple hardware instances and on different surfaces.
With this dataset, we hope to encourage researchers to use the Mini Wheelbot to benchmark their learning-based control methods, even without access to the real hardware.
We include two example implementations of how to use the dataset, i.e., for dynamics learning and state estimation.
A detailled description of this dataset is available in [the brief on arxiv](https://arxiv.org/abs/...).

If you find this dataset helpful, please cite the Mini Wheelbot paper or dataset directly:
```
@inproceedings{hose2025mini,
   title={The {Mini Wheelbot}: A Testbed for Learning-based Balancing, Flips, and Articulated Driving},
   booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
   publisher={IEEE},
   author={Hose, Henrik and Weisgerber, Jan and Trimpe, Sebastian},
   year={2025},
}

@article{hose2026dataset,
  title={The {Mini Wheelbot} Dataset: High-Fidelity Data for Robot Learning
  author={Hose, Henrik and Weisgerber, Jan and Trimpe, Sebastian},
  journal={arXiv preprint arXiv:2502.04582},
  year={2025}
}
```

## 🤖 Overview

The Wheelbot Dataset contains trajectory data from a self-balancing wheeled robot across various experimental conditions including:
- **Pitch experiments**: Random pitch angle tracking
- **Roll experiments**: Random roll angle tracking  
- **Velocity experiments**: Forward/backward velocity tracking
- **Combined experiments**: Velocity + roll, velocity + pitch
- **Yaw experiments**: Random yaw control, circular trajectories, figure-eight patterns

All data is recorded at 1000 Hz and includes:
- IMU data (4x gyroscopes, 4x accelerometers)
- Robot state (yaw, roll, pitch angles and velocities)
- Wheel positions, velocities, and accelerations
- Motor commands
- Setpoints
- Vicon motion capture data (ground truth)
- Battery voltage

## 📦 Installation

### From PyPI (when published)
```bash
pip install wheelbot-dataset
```

### From Source
```bash
git clone https://github.com/wheelbot/dataset.git
cd dataset
pip install -e .
```

### Optional Dependencies
```bash
# For development and examples
pip install wheelbot-dataset[dev]

# For all features (neural network training, etc.)
pip install wheelbot-dataset[all]
```

## 🚀 Quick Start

### 1. Download the Dataset

The dataset is automatically downloadable from Zenodo:

```bash
# Download dataset to ./data directory
python -m wheelbot_dataset download download

# Or download to a custom location
python -m wheelbot_dataset download download --output_dir=my_dataset

# Check if dataset exists
python -m wheelbot_dataset download check
```

From Python:
```python
from wheelbot_dataset import download_dataset, check_dataset

# Download the dataset
download_dataset(output_dir="data")

# Check dataset status
check_dataset("data")
```

### 2. Load and Explore Data

```python
from wheelbot_dataset import Dataset

# Load the dataset
ds = Dataset("data")

# Load a specific experiment group
pitch_group = ds.load_group("pitch")

# Access individual experiments
exp = pitch_group[0]

# View the data
print(exp.data.head())  # Pandas DataFrame
print(exp.meta)         # Metadata dictionary
```

### 3. Process and Filter Data

```python
# Cut off standup and laydown phases
exp_cut = exp.cut_by_condition(
    start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
    end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0
).cut_time(start=2.0, end=2.0)

# Apply filtering
import scipy.signal as signal

def lowpass_filter(df):
    b, a = signal.butter(4, 2*50/1000, btype="low")
    df_filtered = df.copy()
    for col in df.columns:
        df_filtered[col] = signal.filtfilt(b, a, df[col])
    return df_filtered

exp_filtered = exp_cut.apply_filter(lowpass_filter).resample(dt=0.01)
```

### 4. Batch Processing

```python
# Filter entire groups or datasets
group_filtered = pitch_group.map(
    lambda exp: exp
        .filter_by_metadata(experiment_status="success")
        .cut_time(start=2.0, end=2.0)
        .apply_filter(lowpass_filter)
        .resample(dt=0.01)
)

# Filter by metadata
successful_exps = ds.map(
    lambda exp: exp.filter_by_metadata(experiment_status="success")
)
```

## 📊 Visualization

### Plot Time Series

```python
from wheelbot_dataset import plot_timeseries

plot_fields = {
    "Angles": ["/q_yrp/yaw", "/q_yrp/roll", "/q_yrp/pitch"],
    "Angular Velocities": ["/dq_yrp/yaw_vel", "/dq_yrp/roll_vel", "/dq_yrp/pitch_vel"],
    "Motor Commands": ["/tau_DR_command/drive_wheel", "/tau_DR_command/reaction_wheel"],
}

plot_timeseries(
    experiments=exp_filtered,
    groups=plot_fields,
    pdf_path="output.pdf"
)
```

### Plot Histograms

```python
from wheelbot_dataset import plot_histograms

plot_histograms(exp_filtered, "histograms.pdf")
plot_histograms(pitch_group, "group_histograms.pdf")
plot_histograms(ds, "dataset_histograms.pdf")
```

## 🤖 Machine Learning Integration

### Export to NumPy

```python
# Single experiment
columns = ["time"] + list(exp.data.columns)
exp_numpy = exp.to_numpy(columns)

# Entire group
group_numpy = group.map(lambda exp: exp.to_numpy(columns))
```

### Create Training Dataset

```python
from wheelbot_dataset import to_prediction_dataset

fields_states = [
    "/q_yrp/roll", "/q_yrp/pitch",
    "/dq_yrp/yaw_vel", "/dq_yrp/roll_vel", "/dq_yrp/pitch_vel",
    "/dq_DR/drive_wheel", "/dq_DR/reaction_wheel",
]
fields_actions = [
    "/tau_DR_command/drive_wheel", 
    "/tau_DR_command/reaction_wheel",
]

# One-step prediction dataset
states, actions, nextstates, _ = to_prediction_dataset(
    ds_filtered,
    fields_states=fields_states,
    fields_actions=fields_actions,
    N_future=1
)

# Multi-step prediction dataset
states, actions, nextstates, _ = to_prediction_dataset(
    ds_filtered,
    fields_states=fields_states,
    fields_actions=fields_actions,
    N_future=100
)

print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Next states shape: {nextstates.shape}")
```

## 📈 Dataset Statistics

Get comprehensive statistics about your dataset:

```bash
python -m wheelbot_dataset consolidate statistics --dataset_path=data
```

This generates a LaTeX table with:
- Number of trajectories per experiment type
- Total duration (with configurable cutoff)
- Number of crashes/failed experiments

### Analyze Update Rates

Check the actual update rates of different sensor signals:

```bash
python -m wheelbot_dataset consolidate updaterates \
    --dataset_path=data \
    --group_name=pitch \
    --index=4
```

## 🗂️ Dataset Structure

```
data/
├── pitch/              # Pitch angle tracking experiments
│   ├── 0.csv
│   ├── 0.meta
│   ├── 0.mp4
│   ├── ...
├── roll/               # Roll angle tracking experiments
├── roll_max/           # Maximum roll angle experiments
├── velocity/           # Forward/backward velocity
├── velocity_pitch/     # Combined velocity and pitch
├── velocity_roll/      # Combined velocity and roll
├── yaw/                # Random yaw control
├── yaw_circle/         # Circular trajectories
└── yaw_figure_eight/   # Figure-eight trajectories
```

Each experiment includes:
- `.csv` - Time series data at 1000 Hz
- `.meta` - JSON metadata (robot ID, surface, experiment status)
- `.mp4` - Video recording
- `.log` - Raw log file
- `.pkl` - Setpoint data
- `.preview.pdf` - Quick visualization
- `.setpoints.pdf` - Setpoint visualization

## 🎓 Examples

Comprehensive examples are available in the `examples/` directory:

### Estimation (`examples/estimator/`)
- Complementary filter implementations
- Orientation estimation
- Comparison between onboard estimation and Vicon ground truth

### Model Learning (`examples/modellearning_nn/`)
- Neural network-based dynamics learning
- One-step and multi-step prediction models
- Export trained models to CasADi for MPC
- Example MPC implementation

### System Identification (`examples/sysid/`)
- Physics-based model identification
- Equations of motion derivation
- Parameter estimation from data

### Basic Usage (`examples/basic_usage.py`)

A comprehensive example demonstrating the main features of the package:

```bash
# Run the basic usage example
python examples/basic_usage.py
```

The example script shows:
- Loading datasets and experiment groups
- Cutting and filtering data
- Applying custom filters (e.g., lowpass)
- Resampling data
- Batch processing with map operations
- Plotting time series and histograms
- Exporting to NumPy arrays
- Creating training datasets for machine learning

##  Data Fields

### IMU Data
- `/gyro{0-3}/{x,y,z}` - Angular velocities from 4 gyroscopes
- `/accel{0-3}/{x,y,z}` - Linear accelerations from 4 accelerometers

### Robot State
- `/q_yrp/{yaw,roll,pitch}` - Euler angles (rad)
- `/dq_yrp/{yaw_vel,roll_vel,pitch_vel}` - Angular velocities (rad/s)
- `/q_DR/{drive_wheel,reaction_wheel}` - Wheel positions (rad)
- `/dq_DR/{drive_wheel,reaction_wheel}` - Wheel velocities (rad/s)
- `/ddq_DR/{drive_wheel,reaction_wheel}` - Wheel accelerations (rad/s²)

### Commands
- `/tau_DR_command/{drive_wheel,reaction_wheel}` - Motor torque commands (N⋅m)

### Setpoints
- `/setpoint/{yaw,roll,pitch}` - Target Euler angles
- `/setpoint/{yaw_rate,roll_rate,pitch_rate}` - Target angular velocities
- `/setpoint/driving_wheel_angle` - Target wheel angle
- `/setpoint/driving_wheel_angular_velocity` - Target wheel velocity

### Motion Capture
- `/vicon_position/{x,y,z}` - Position from Vicon (m)
- `/vicon_orientation_wxyz/{w,x,y,z}` - Quaternion orientation

### Other
- `battery/voltage` - Battery voltage (V)

## 🛠️ Recording New Data

For information on recording new trajectories, see [README_RECORDING.md](README_RECORDING.md).

## 📚 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{wheelbot-dataset-2026,
  title={Wheelbot Dataset: A Comprehensive Dataset for Self-Balancing Robot Research},
  author={Wheelbot Team},
  year={2026},
  publisher={Zenodo},
  howpublished={\url{https://zenodo.org/record/XXXXXX}}
}
```
