# Complementary Filter for Orientation Estimation

This directory contains examples for orientation estimation using a complementary filter that fuses IMU (gyroscope and accelerometer) measurements with motor encoder data.

## Overview

The complementary filter estimates the robot's orientation (Yaw-Pitch-Roll) and angular velocities by combining:
- **Gyroscope measurements**: High-frequency orientation tracking
- **Accelerometer measurements**: Gravity-based roll and pitch correction
- **Motor encoder data**: Compensation for robot's rotational dynamics

This is a Python implementation of the C++ Estimator used onboard the wheelbot.

## Main File

**`complementary_filter.py`** - Complete implementation with Estimator class and plotting utilities
- `Estimator` class: Complementary filter implementation
- `quaternion_to_euler_zxy()`: Convert Vicon quaternions to Euler angles
- `run_filter_on_experiment()`: Apply filter to experiment data
- `plot_filter_results()`: Visualization with Vicon ground truth comparison
- `main()`: Example script to process experiments and generate plots

## Usage

### Basic Example

Run the complementary filter on experiments and generate plots:

```bash
python complementary_filter.py
```

This will:
- Process experiments from yaw, pitch, and roll groups
- Generate comparison plots showing:
  - Filter estimates (complementary filter)
  - Onboard estimates (if available)
  - Vicon ground truth (if available)
- Save results to `plots/filter_estimation_{group}.pdf`

**Vicon Ground Truth**: When Vicon data is available in the experiment, it will be automatically detected and plotted alongside the filter estimates. The yaw offset is automatically aligned for comparison.

### Using the Filter in Your Code

```python
from complementary_filter import Estimator, run_filter_on_experiment
from wheelbot_dataset import Dataset

# Load dataset
ds = Dataset("../../data")
exp = ds.groups["yaw"].experiments[0]

# Run filter on experiment
results = run_filter_on_experiment(exp)

# Access results
time = results['time']
yaw_estimated = results['yaw']
pitch_estimated = results['pitch']
roll_estimated = results['roll']
```

### Advanced Usage: Custom Filtering with Vicon Comparison

```python
from complementary_filter import Estimator, run_filter_on_experiment, plot_filter_results, quaternion_to_euler_zxy
from wheelbot_dataset import Dataset
import numpy as np

# Load dataset
ds = Dataset("../../data")
exp = ds.groups["yaw"].experiments[0]

# Run filter on experiment
filter_results = run_filter_on_experiment(exp)
time = filter_results['time']

# Extract Vicon ground truth if available
vicon_data = None
vicon_quat_fields = [
    "/vicon_orientation_wxyz/w",
    "/vicon_orientation_wxyz/x", 
    "/vicon_orientation_wxyz/y",
    "/vicon_orientation_wxyz/z"
]

if all(f in exp.data.columns for f in vicon_quat_fields):
    df = exp.data
    mask = (df.index >= time[0]) & (df.index <= time[-1])
    df_sliced = df[mask]
    
    qw = df_sliced["/vicon_orientation_wxyz/w"].to_numpy()
    qx = df_sliced["/vicon_orientation_wxyz/x"].to_numpy()
    qy = df_sliced["/vicon_orientation_wxyz/y"].to_numpy()
    qz = df_sliced["/vicon_orientation_wxyz/z"].to_numpy()
    
    yaw_vicon, roll_vicon, pitch_vicon = quaternion_to_euler_zxy(qw, qx, qy, qz)
    vicon_data = {'yaw': yaw_vicon, 'roll': roll_vicon, 'pitch': pitch_vicon}

# Extract onboard estimates
onboard_data = None
if all(f'/q_yrp/{a}' in exp.data.columns for a in ['yaw', 'roll', 'pitch']):
    mask = (df.index >= time[0]) & (df.index <= time[-1])
    df_sliced = df[mask]
    onboard_data = {
        'yaw': df_sliced['/q_yrp/yaw'].to_numpy(),
        'roll': df_sliced['/q_yrp/roll'].to_numpy(),
        'pitch': df_sliced['/q_yrp/pitch'].to_numpy()
    }

# Plot with all comparisons
fig = plot_filter_results(
    time,
    filter_results,
    vicon_data=vicon_data,
    onboard_data=onboard_data,
    title="Complete Comparison"
)
```

## Algorithm Details

### Complementary Filter

The complementary filter combines gyroscope and accelerometer measurements:

1. **Gyroscope Integration**: Integrates angular velocities for orientation estimate
   - High-frequency tracking
   - Prone to drift over time

2. **Accelerometer Correction**: Uses gravity vector for roll/pitch estimation
   - Low-frequency correction
   - Immune to gyroscope drift
   - Cannot estimate yaw (no magnetic reference)

3. **Fusion**: Weighted combination with gain parameters (K_a_P, K_a_R)
   ```
   roll_estimate = (1 - K_a_R) * roll_gyro + K_a_R * roll_accel
   pitch_estimate = (1 - K_a_P) * pitch_gyro + K_a_P * pitch_accel
   yaw_estimate = yaw_gyro (no correction available)
   ```

### IMU Configuration

The wheelbot uses 4 IMUs positioned along the robot's length:
- IMUs 0-1: One side (with R01 rotation matrix)
- IMUs 2-3: Other side (with R23 rotation matrix)
- Position offsets (X1): Used for acceleration compensation

### Coordinate Frames

- **Body Frame**: Fixed to robot body
- **Sensor Frames**: Individual IMU orientations
- **World Frame**: Fixed inertial reference
- Rotation matrices (R_Bi) transform sensor measurements to body frame

## Output Files

### Plots
- `plots/filter_estimation_{group}.pdf` - Multi-page PDF with estimation results
  - One page per experiment
  - 3 subplots: Roll, Pitch, Yaw vs time
  - Comparison lines:
    - **Blue solid**: Complementary filter estimate (this implementation)
    - **Orange dashed**: Onboard estimate (if available in dataset)
    - **Green dotted**: Vicon ground truth (if available in dataset)
  - Yaw offset automatically aligned between filter and Vicon for comparison

## Parameters

### Filter Gains
- `alpha = 0.01`: Complementary filter fusion parameter (default)
  - Controls blend between gyro and accelerometer estimates
  - Lower values → more gyroscope influence (smoother, but more drift)
  - Higher values → more accelerometer influence (better drift correction, more noise)

### Physical Constants
- `g = 9.81`: Gravitational acceleration (m/s²)
- `r = 32e-3`: Wheel radius (meters)
- `X1`: IMU positions along robot length (meters)
- `dt = 1e-3`: Default timestep (1ms)

## Notes

- The filter assumes small angles for accelerometer-based orientation estimation
- Yaw estimation relies solely on gyroscope integration (no magnetic reference)
- For long experiments, yaw drift will accumulate
- The upside-down mode (`upside_down` flag) handles inverted robot orientation
