# Recording New Trajectories

This guide explains how to record new experimental data with the Wheelbot robot using the `wheelbot-dataset` package.

## 🎬 Prerequisites

### Hardware Requirements
- Mini Wheelbot robot with:
  - 4 IMUs (gyroscopes and accelerometers)
  - 2 motorized wheels (drive wheel and reaction wheel)
  - Onboard computer with SSH access
  - Charged battery
- Vicon motion capture system (optional but recommended for ground truth)
- Video camera or webcam for recording (optional)

### Software Requirements
- Python 3.10 or higher
- wheelbot-dataset package installed
- SSH access to the robot
- Network connection to robot

### Network Setup
- Robot must be accessible via SSH
- Default robot names: `wheelbot-beta-1`, `wheelbot-beta-2`, `wheelbot-beta-3`, etc.
- Video device (if using): typically `/dev/video4` or `/dev/video0`

## 🚀 Quick Start

### Basic Recording Commands

Each experiment type has a dedicated command. The simplest way to record is:

```bash
# Record pitch experiments
python -m wheelbot_dataset.record pitch

# Record roll experiments
python -m wheelbot_dataset.record roll

# Record velocity experiments
python -m wheelbot_dataset.record vel

# Record yaw experiments
python -m wheelbot_dataset.record yaw
```

### Specifying Robot and Surface

```bash
python -m wheelbot_dataset.record pitch \
    --wheelbot_name="wheelbot-beta-2" \
    --surface="black_pvc" \
    --video_device="/dev/video4"
```

## 📋 Available Experiment Types

### 1. Pitch Experiments
Random pitch angle tracking with pseudo-random binary sequences (PRBS):
```bash
python -m wheelbot_dataset.record pitch
```
- **Duration**: ~30 seconds per trial
- **Range**: 10-20 degrees
- **Trials**: 5 different pitch ranges

### 2. Roll Experiments
Random roll angle tracking:
```bash
python -m wheelbot_dataset.record roll
```
- **Duration**: ~30 seconds per trial
- **Range**: 4-10 degrees
- **Trials**: 5 different roll ranges

### 3. Roll Max Experiments
Maximum roll angle experiments at specific angle:
```bash
python -m wheelbot_dataset.record roll_max --angle=10
```
- **Duration**: ~30 seconds per trial
- **Default angle**: 10 degrees
- **Trials**: 10 repetitions

### 4. Velocity Experiments
Forward/backward velocity tracking:
```bash
python -m wheelbot_dataset.record vel
```
- **Duration**: ~10 seconds per trial
- **Range**: 0.1-1.0 m/s
- **Trials**: 5 different velocity ranges

### 5. Yaw Experiments
Random yaw control with PRBS:
```bash
python -m wheelbot_dataset.record yaw --yaw_range_deg=90 --duration_s=30
```
- **Default range**: 90 degrees
- **Default duration**: 30 seconds
- **Trials**: 10 with different random seeds

### 6. Circular Trajectories
Yaw tracking following circular paths:
```bash
python -m wheelbot_dataset.record yaw_figure_circle
```
- **Duration**: Variable based on circle size
- **Radius**: Multiple circle radii tested
- **Trials**: Multiple repetitions

### 7. Figure-Eight Trajectories
Yaw tracking following figure-eight patterns:
```bash
python -m wheelbot_dataset.record yaw_figure_eight
```
- **Duration**: Variable based on pattern size
- **Pattern**: Figure-eight with different sizes
- **Trials**: Multiple repetitions

### 8. Combined Experiments

#### Velocity + Roll + Pitch
```bash
python -m wheelbot_dataset.record velrollpitch
```
Combined excitation of all three degrees of freedom.

#### Linear Velocity
Simple linear acceleration/deceleration:
```bash
python -m wheelbot_dataset.record lin
```

#### Linear with Lean
Linear velocity with pitch lean during acceleration:
```bash
python -m wheelbot_dataset.record linwithlean
```

## 🎯 Recording Workflow

### Interactive Mode

When you run a recording command, you'll be prompted for each trial:

1. **Preview**: A plot shows the planned setpoint trajectory
2. **Confirmation**: You can choose to:
   - **Continue** (press Enter or 'c'): Record the experiment
   - **Skip** (press 's'): Skip this trial
   - **Abort** (press 'a'): Stop the entire recording session

3. **Recording**: The robot executes the trajectory
4. **Post-processing**: Data is automatically saved and visualized

### Example Session

```bash
$ python -m wheelbot_dataset.record pitch

Pitch: 10.0
[Preview window shows trajectory]
Continue, Skip, or Abort? (Enter/c/s/a): 

# Press Enter to continue
Recording experiment to data/pitch/0
[Robot executes trajectory]
✓ Saved: data/pitch/0.csv
✓ Saved: data/pitch/0.meta
✓ Saved: data/pitch/0.mp4
✓ Generated: data/pitch/0.preview.pdf

Pitch: 12.5
[Preview window shows trajectory]
Continue, Skip, or Abort? (Enter/c/s/a): s

# Skipping to next...
```

## 📁 Output Files

Each recorded experiment generates the following files:

- **`{n}.csv`**: Time series data at 1000 Hz
- **`{n}.meta`**: JSON metadata including:
  - `experiment_status`: "success" or "failed"
  - `wheelbot`: Robot identifier
  - `surface`: Surface type
  - `uuid`: Unique experiment identifier
- **`{n}.mp4`**: Video recording (if camera enabled)
- **`{n}.log`**: Raw robot log file
- **`{n}.pkl`**: Setpoint data (Python pickle)
- **`{n}.preview.pdf`**: Quick visualization of recorded data
- **`{n}.setpoints.pdf`**: Visualization of commanded setpoints

Files are automatically numbered sequentially (0, 1, 2, ...) in each experiment group folder.

## ⚙️ Advanced Configuration

### Custom Robot Parameters

```bash
python -m wheelbot_dataset.record pitch \
    --wheelbot_name="wheelbot-custom" \
    --surface="carpet" \
    --video_device="/dev/video0"
```

### Batch Recording with Different Seeds

For reproducibility and variety, use different random seeds:

```python
# In wheelbot_dataset/record.py, modify:
global_seed_offset = 0  # Change to 50, 100, 150, etc. for different robots
```

Then record with multiple robots:
```bash
# Robot 1 (seed offset 0)
python -m wheelbot_dataset.record pitch

# Robot 2 (seed offset 50)
python -m wheelbot_dataset.record pitch

# Robot 3 (seed offset 100)
python -m wheelbot_dataset.record pitch
```

### Disabling Video Recording

If no camera is available:
```bash
python -m wheelbot_dataset.record pitch --video_device=None
```

### Custom Setpoint Parameters

Edit `wheelbot_dataset/record.py` to customize:
- Trajectory duration
- Stabilization intervals
- Random seed offsets
- Signal ranges and slew rates
- Perturbation standard deviations

Example for velocity experiments:
```python
velocity, roll, pitch = generate_setpoints(
    duration_s=20,                    # Custom duration
    stabilize_every_n_seconds=10,
    stabilize_for=5,
    dt=0.05,
    random_seed=seed + global_seed_offset,
    roll_range_deg=0,
    pitch_range_deg=0,
    vel_range=1.5,                    # Custom velocity range
    vel_slew_per_0p2s=0.5,
    roll_pert_stddev=0,
    pitch_pert_stddev=0,
    vel_pert_stddev=0
)
```

## 🔧 Troubleshooting

### Robot Not Responding
- Check SSH connection: `ssh wheelbot-beta-X`
- Verify robot is powered on
- Check network connectivity
- Ensure robot control software is running

### Video Recording Fails
- Check video device exists: `ls /dev/video*`
- Try different device: `--video_device="/dev/video0"`
- Disable video if not needed: `--video_device=None`

### Experiment Crashes
- Experiments are automatically marked as `experiment_status: "failed"` in metadata
- Failed experiments can be filtered out during analysis
- Review setpoint parameters if crashes are frequent
- Check battery voltage (low battery can cause failures)

### Missing Vicon Data
- Ensure Vicon system is running
- Check robot is visible to cameras
- Vicon data is optional; experiments can run without it

## 📊 Quality Control

### After Recording

1. **Check experiment status**:
```bash
python -m wheelbot_dataset consolidate statistics --dataset_path=data
```

2. **Review preview PDFs**: Quickly scan `{n}.preview.pdf` files

3. **Filter failed experiments**:
```python
from wheelbot_dataset import Dataset

ds = Dataset("data")
ds_filtered = ds.map(
    lambda exp: exp.filter_by_metadata(experiment_status="success")
)
```

### Marking Failed Experiments

If you notice an experiment failed after recording:
```python
import json

# Edit the .meta file
with open("data/pitch/5.meta", "r") as f:
    meta = json.load(f)

meta["experiment_status"] = "failed"

with open("data/pitch/5.meta", "w") as f:
    json.dump(meta, f, indent=2)
```

## 🗂️ Data Organization

### Recommended Structure

```
project/
├── 20260112_data_beta_1/    # Recording session 1
│   ├── pitch/
│   ├── roll/
│   └── ...
├── 20260112_data_beta_3/    # Recording session 2
│   ├── pitch/
│   ├── roll/
│   └── ...
└── data/                     # Consolidated dataset
    ├── pitch/
    ├── roll/
    └── ...
```

### Consolidating Multiple Sessions

After recording with multiple robots or sessions:

```bash
python -m wheelbot_dataset consolidate consolidate \
    --input_dataset_paths="['20260112_data_beta_1','20260112_data_beta_3']" \
    --output_dataset_path="data"
```

This will:
- Combine all experiments from multiple sources
- Renumber experiments sequentially
- Verify all required files exist
- Create a unified dataset

## 🔧 Dataset Management

### Consolidate Multiple Datasets

If you've recorded data across multiple sessions or robots:

```bash
python -m wheelbot_dataset consolidate consolidate \
    --input_dataset_paths="['session1','session2','session3']" \
    --output_dataset_path="data_consolidated"
```

The consolidation tool will:
- Merge all experiments from the input datasets
- Renumber experiments sequentially within each group (starting from 0)
- Verify all required files exist for each experiment
- Skip incomplete experiments (missing files)
- Create a unified, clean dataset

### Check Dataset Integrity

After recording or consolidating, verify your dataset:

```python
from wheelbot_dataset import Dataset

ds = Dataset("data")
for group_name, group in ds.groups.items():
    print(f"{group_name}: {len(group.experiments)} experiments")
```

Or use the built-in statistics tool:

```bash
python -m wheelbot_dataset consolidate statistics --dataset_path=data
```

This provides detailed statistics including:
- Number of trajectories per group
- Total duration (with adjustable cutoff for standup/laydown)
- Number and percentage of failed experiments
- LaTeX-formatted table for publication

### Dataset Quality Metrics

Monitor data quality with the update rate analyzer:

```bash
python -m wheelbot_dataset consolidate updaterates \
    --dataset_path=data \
    --group_name=pitch \
    --index=4
```

This shows the actual sampling rates of different sensors, helping identify:
- Low-rate signals that may need attention
- Sensors not updating as expected
- Data quality issues

## 🎓 Best Practices

1. **Pre-flight Checklist**:
   - Battery fully charged
   - Robot on appropriate surface
   - Vicon system calibrated (if using)
   - Camera working (if using)
   - SSH connection stable

2. **During Recording**:
   - Monitor robot behavior
   - Skip trials if robot is unstable
   - Take breaks to prevent overheating
   - Check battery voltage regularly

3. **Post-recording**:
   - Review all preview PDFs
   - Run statistics to check crash rates
   - Back up raw data before processing
   - Mark any obviously failed experiments

4. **Multiple Robots**:
   - Use different `global_seed_offset` values
   - Keep consistent naming conventions
   - Record session metadata (date, robot ID, conditions)

5. **Documentation**:
   - Note any unusual conditions
   - Document hardware changes
   - Keep log of recording sessions
   - Track battery replacements

## 📝 Example Recording Script

For automated batch recording:

```python
#!/usr/bin/env python3
"""Batch recording script"""

import subprocess

# Configuration
robot_name = "wheelbot-beta-2"
surface = "black_pvc"
video_device = "/dev/video4"

experiments = [
    "pitch",
    "roll", 
    "roll_max",
    "vel",
    "yaw",
    "yaw_figure_circle",
    "yaw_figure_eight",
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Recording: {exp}")
    print('='*60)
    
    cmd = [
        "python", "-m", "wheelbot_dataset.record", exp,
        f"--wheelbot_name={robot_name}",
        f"--surface={surface}",
        f"--video_device={video_device}"
    ]
    
    subprocess.run(cmd)
    
    input("Press Enter to continue to next experiment...")

print("\nAll experiments completed!")
```

## 🚦 Safety Notes

- Always supervise the robot during experiments
- Ensure adequate space for robot movement
- Keep emergency stop accessible
- Monitor battery temperature
- Stop if robot shows unusual behavior

## 📞 Support

For issues or questions about recording:
- Check existing experiment examples in `data/`
- Review `wheelbot_dataset/record.py` for implementation details
- Consult `wheelbot_dataset/recording/experiment.py` for low-level recording functions
- Open an issue on GitHub for bugs or feature requests
