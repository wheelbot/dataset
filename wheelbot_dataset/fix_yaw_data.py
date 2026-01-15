"""
Fix yaw data in the yaw_circle experiments using the complementary filter.

This module processes CSV files in yaw-related experiment groups, applies the 
complementary filter to estimate orientation, and updates the q_yrp and dq_yrp 
fields with the filter output. The original files are backed up with .backup extension.
"""

import os
import shutil
import glob
import numpy as np
import pandas as pd

# Import the Estimator from the examples/estimator module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'estimator'))
from complementary_filter import Estimator


def fix_experiment_yaw(csv_path: str, meta_path: str = None) -> bool:
    """
    Fix the yaw data for a single experiment CSV file.
    
    Args:
        csv_path: Path to the CSV file
        meta_path: Path to the metadata file (optional, will be inferred if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    # Infer meta path if not provided
    if meta_path is None:
        base_path = csv_path.rsplit('.', 1)[0]
        meta_path = base_path + '.meta'
    
    # Check if meta file exists
    if not os.path.exists(meta_path):
        print(f"  Warning: No metadata file found at {meta_path}, creating minimal metadata")
        # Create minimal metadata for the experiment
        meta_data = {}
    
    print(f"  Loading data from {csv_path}")
    
    # Load the CSV data directly
    df = pd.read_csv(csv_path)
    if "_time" in df.columns:
        df = df.set_index("_time")
    
    # Check if all required fields are present
    gyro_fields = [f"/gyro{i}/{axis}" for i in range(4) for axis in ['x', 'y', 'z']]
    accel_fields = [f"/accel{i}/{axis}" for i in range(4) for axis in ['x', 'y', 'z']]
    motor_fields = ['/q_DR/drive_wheel', '/dq_DR/drive_wheel', '/ddq_DR/drive_wheel']
    
    required_fields = gyro_fields + accel_fields + motor_fields
    missing_fields = [f for f in required_fields if f not in df.columns]
    
    if missing_fields:
        print(f"    Missing fields: {missing_fields[:5]}...")
        return False
    
    # Initialize estimator
    estimator = Estimator(N_IMUS=4, N_MOTORS=2)
    
    # Prepare data storage
    n_samples = len(df)
    
    # Results storage
    yaw = np.zeros(n_samples)
    roll = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    yaw_vel = np.zeros(n_samples)
    roll_vel = np.zeros(n_samples)
    pitch_vel = np.zeros(n_samples)
    
    print(f"  Processing {n_samples} samples...")
    
    # Process each timestep
    for i in range(n_samples):
        # Extract gyro measurements (3x4 matrix)
        omega_B = np.zeros((3, 4), dtype=np.float32)
        for j in range(4):
            omega_B[0, j] = df[f'/gyro{j}/x'].iloc[i]
            omega_B[1, j] = df[f'/gyro{j}/y'].iloc[i]
            omega_B[2, j] = df[f'/gyro{j}/z'].iloc[i]
        
        # Extract accel measurements (3x4 matrix)
        a_B = np.zeros((3, 4), dtype=np.float32)
        for j in range(4):
            a_B[0, j] = df[f'/accel{j}/x'].iloc[i]
            a_B[1, j] = df[f'/accel{j}/y'].iloc[i]
            a_B[2, j] = df[f'/accel{j}/z'].iloc[i]
        
        # Extract motor states (3x2 matrix: position, velocity, acceleration for 2 motors)
        motor_states = np.zeros((3, 2), dtype=np.float32)
        motor_states[0, 0] = df['/q_DR/drive_wheel'].iloc[i]
        motor_states[1, 0] = df['/dq_DR/drive_wheel'].iloc[i]
        motor_states[2, 0] = df['/ddq_DR/drive_wheel'].iloc[i]
        
        # Reaction wheel
        if '/q_DR/reaction_wheel' in df.columns:
            motor_states[0, 1] = df['/q_DR/reaction_wheel'].iloc[i]
            motor_states[1, 1] = df['/dq_DR/reaction_wheel'].iloc[i]
            motor_states[2, 1] = df['/ddq_DR/reaction_wheel'].iloc[i]
        else:
            motor_states[:, 1] = motor_states[:, 0]
        
        # Update estimator
        state = estimator.update(omega_B, a_B, motor_states)
        
        # Store results - state returns: [yaw, roll, pitch, yaw_vel, roll_vel, pitch_vel, ...]
        yaw[i] = state[0]
        roll[i] = state[1]
        pitch[i] = state[2]
        yaw_vel[i] = state[3]
        roll_vel[i] = state[4]
        pitch_vel[i] = state[5]
    
    # Create backup of original file
    backup_path = csv_path + '.backup'
    if not os.path.exists(backup_path):
        print(f"  Creating backup at {backup_path}")
        shutil.copy2(csv_path, backup_path)
    else:
        print(f"  Backup already exists at {backup_path}")
    
    # Update the DataFrame with the new values
    # Reset index to access _time column
    df_with_time = df.reset_index()
    
    # Update q_yrp fields
    df_with_time['/q_yrp/yaw'] = yaw
    df_with_time['/q_yrp/roll'] = roll
    df_with_time['/q_yrp/pitch'] = pitch
    
    # Update dq_yrp fields
    df_with_time['/dq_yrp/yaw_vel'] = yaw_vel
    df_with_time['/dq_yrp/roll_vel'] = roll_vel
    df_with_time['/dq_yrp/pitch_vel'] = pitch_vel
    
    # Save the updated data
    print(f"  Saving updated data to {csv_path}")
    df_with_time.to_csv(csv_path, index=False)
    
    return True


def fix_yaw_data_for_dataset(data_base_dir: str, folders_to_process: list[str] = None) -> tuple[int, int]:
    """
    Process all experiments in specified yaw-related folders.
    
    Args:
        data_base_dir: Path to the data directory containing experiment folders
        folders_to_process: List of folder names to process. If None, defaults to
                          ['yaw_circle', 'yaw_figure_eight', 'yaw', 'yaw_human']
    
    Returns:
        Tuple of (total_success, total_failed) counts
    """
    if folders_to_process is None:
        folders_to_process = ['yaw_circle', 'yaw_figure_eight', 'yaw', 'yaw_human']
    
    # Process each folder
    total_success = 0
    total_failed = 0
    
    for folder_name in folders_to_process:
        data_dir = os.path.join(data_base_dir, folder_name)
        
        print(f"\n{'=' * 50}")
        print(f"Processing folder: {folder_name}")
        print(f"Path: {data_dir}")
        print('=' * 50)
        
        if not os.path.exists(data_dir):
            print(f"  Warning: Data directory not found: {data_dir}")
            print(f"  Skipping {folder_name}\n")
            continue
        
        # Find all CSV files in the directory
        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        
        # Filter out backup files
        csv_files = [f for f in csv_files if not f.endswith('.backup')]
        
        if not csv_files:
            print(f"  No CSV files found in {folder_name}")
            continue
        
        print(f"  Found {len(csv_files)} CSV files to process\n")
        
        # Process each CSV file
        success_count = 0
        failed_count = 0
        
        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            print(f"  Processing: {filename}")
            
            try:
                if fix_experiment_yaw(csv_path):
                    success_count += 1
                    print(f"    ✓ Successfully updated {filename}\n")
                else:
                    failed_count += 1
                    print(f"    ✗ Failed to process {filename}\n")
            except Exception as e:
                failed_count += 1
                print(f"    ✗ Error processing {filename}: {e}\n")
        
        # Folder summary
        print(f"\n  {folder_name} Summary:")
        print(f"    Successful: {success_count}")
        print(f"    Failed: {failed_count}")
        
        total_success += success_count
        total_failed += failed_count
    
    # Overall summary
    print("\n" + "=" * 50)
    print(f"Overall Processing Complete!")
    print(f"  Total Successful: {total_success}")
    print(f"  Total Failed: {total_failed}")
    print("=" * 50)
    
    return total_success, total_failed


def main():
    """Main entry point for command-line usage."""
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.join(script_dir, '..', 'data')
    data_base_dir = os.path.normpath(data_base_dir)
    
    fix_yaw_data_for_dataset(data_base_dir)


if __name__ == "__main__":

    for folder_name in ["data_felt", "data"]:
        main(folder_name)
