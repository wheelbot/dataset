"""
Plot comparison of yaw data before and after fixing.

This script creates plots comparing:
1. RPY from Vicon (ground truth)
2. RPY from the updated CSV (after fix_yaw_data.py)
3. RPY from the backup CSV (original data)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import quaternion conversion from filter_complementary
from filter_complementary import quaternion_to_euler_zxy


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load CSV file and set time column as index if present."""
    df = pd.read_csv(csv_path)
    if "_time" in df.columns:
        df = df.set_index("_time")
    return df


def plot_comparison(csv_path: str, backup_path: str, output_dir: str) -> bool:
    """
    Create comparison plot for a single experiment.
    
    Args:
        csv_path: Path to the updated CSV file
        backup_path: Path to the backup (original) CSV file
        output_dir: Directory to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    filename = os.path.basename(csv_path)
    exp_name = filename.rsplit('.', 1)[0]
    
    print(f"  Loading updated data from {csv_path}")
    df_updated = load_csv_data(csv_path)
    
    print(f"  Loading original data from {backup_path}")
    df_original = load_csv_data(backup_path)
    
    # Get time array
    time = df_updated.index.to_numpy()
    
    # Extract updated q_yrp data
    yaw_updated = df_updated['/q_yrp/yaw'].to_numpy()
    roll_updated = df_updated['/q_yrp/roll'].to_numpy()
    pitch_updated = df_updated['/q_yrp/pitch'].to_numpy()
    
    # Extract original q_yrp data
    yaw_original = df_original['/q_yrp/yaw'].to_numpy()
    roll_original = df_original['/q_yrp/roll'].to_numpy()
    pitch_original = df_original['/q_yrp/pitch'].to_numpy()
    
    # Extract Vicon data if available
    vicon_quat_fields = [
        "/vicon_orientation_wxyz/w",
        "/vicon_orientation_wxyz/x",
        "/vicon_orientation_wxyz/y",
        "/vicon_orientation_wxyz/z"
    ]
    
    has_vicon = all(f in df_updated.columns for f in vicon_quat_fields)
    
    if has_vicon:
        qw = df_updated["/vicon_orientation_wxyz/w"].to_numpy()
        qx = df_updated["/vicon_orientation_wxyz/x"].to_numpy()
        qy = df_updated["/vicon_orientation_wxyz/y"].to_numpy()
        qz = df_updated["/vicon_orientation_wxyz/z"].to_numpy()
        
        # Check if vicon data is valid (not all zeros)
        if np.all(qw == 0) and np.all(qx == 0) and np.all(qy == 0) and np.all(qz == 0):
            print(f"    Warning: Vicon data is all zeros")
            has_vicon = False
        else:
            yaw_vicon, roll_vicon, pitch_vicon = quaternion_to_euler_zxy(qw, qx, qy, qz)
    
    # Find region of interest (where actions are nonzero) for better visualization
    if '/tau_DR_command/reaction_wheel' in df_updated.columns:
        action_data = df_updated['/tau_DR_command/reaction_wheel'].to_numpy()
        nonzero_idx = np.where(np.abs(action_data) > 0)[0]
        
        if len(nonzero_idx) > 0:
            # Add some margin
            dt_sample = time[1] - time[0] if len(time) > 1 else 0.001
            margin_samples = int(2.0 / dt_sample)
            
            start_idx = max(0, nonzero_idx[0] - margin_samples)
            end_idx = min(len(time) - 1, nonzero_idx[-1] + margin_samples)
        else:
            start_idx = 0
            end_idx = len(time) - 1
    else:
        start_idx = 0
        end_idx = len(time) - 1
    
    # Cut data
    time_cut = time[start_idx:end_idx+1]
    yaw_updated_cut = yaw_updated[start_idx:end_idx+1]
    roll_updated_cut = roll_updated[start_idx:end_idx+1]
    pitch_updated_cut = pitch_updated[start_idx:end_idx+1]
    yaw_original_cut = yaw_original[start_idx:end_idx+1]
    roll_original_cut = roll_original[start_idx:end_idx+1]
    pitch_original_cut = pitch_original[start_idx:end_idx+1]
    
    if has_vicon:
        yaw_vicon_cut = yaw_vicon[start_idx:end_idx+1]
        roll_vicon_cut = roll_vicon[start_idx:end_idx+1]
        pitch_vicon_cut = pitch_vicon[start_idx:end_idx+1]
        
        # Align vicon yaw to match updated estimate (they may have different offsets)
        yaw_vicon_cut = yaw_vicon_cut - np.mean(yaw_vicon_cut) + np.mean(yaw_updated_cut)
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Yaw Fix Comparison: yaw_circle/{exp_name}", fontsize=14)
    
    # Roll
    axes[0].plot(time_cut, roll_updated_cut, label='Updated (Filter)', linewidth=1.5, color='blue')
    axes[0].plot(time_cut, roll_original_cut, label='Original (Onboard)', linewidth=1.0, linestyle='--', color='orange', alpha=0.8)
    if has_vicon:
        axes[0].plot(time_cut, roll_vicon_cut, label='Vicon (Ground Truth)', linewidth=1.5, linestyle=':', color='green', alpha=0.8)
    axes[0].set_ylabel('Roll [rad]')
    axes[0].set_xlabel('Time [s]')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Roll Angle')
    
    # Pitch
    axes[1].plot(time_cut, pitch_updated_cut, label='Updated (Filter)', linewidth=1.5, color='blue')
    axes[1].plot(time_cut, pitch_original_cut, label='Original (Onboard)', linewidth=1.0, linestyle='--', color='orange', alpha=0.8)
    if has_vicon:
        axes[1].plot(time_cut, pitch_vicon_cut, label='Vicon (Ground Truth)', linewidth=1.5, linestyle=':', color='green', alpha=0.8)
    axes[1].set_ylabel('Pitch [rad]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Pitch Angle')
    
    # Yaw
    axes[2].plot(time_cut, yaw_updated_cut, label='Updated (Filter)', linewidth=1.5, color='blue')
    axes[2].plot(time_cut, yaw_original_cut, label='Original (Onboard)', linewidth=1.0, linestyle='--', color='orange', alpha=0.8)
    if has_vicon:
        axes[2].plot(time_cut, yaw_vicon_cut, label='Vicon (Ground Truth)', linewidth=1.5, linestyle=':', color='green', alpha=0.8)
    axes[2].set_ylabel('Yaw [rad]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Yaw Angle')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"fix_yaw_comparison_{exp_name}.pdf")
    plt.savefig(output_path)
    plt.close()
    
    print(f"    Saved: {output_path}")
    return True


def main():
    """Create comparison plots for all yaw_circle experiments."""
    
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data', 'yaw_circle')
    data_dir = os.path.normpath(data_dir)
    
    # Create output directory
    output_dir = os.path.join(script_dir, 'plots', 'fix_yaw')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing experiments in: {data_dir}")
    print(f"Output directory: {output_dir}\n")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Find all CSV files in the yaw_circle directory (excluding backups)
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    csv_files = [f for f in csv_files if not f.endswith('.backup')]
    
    if not csv_files:
        print("No CSV files found in the yaw_circle directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    # Process each CSV file
    success_count = 0
    failed_count = 0
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        backup_path = csv_path + '.backup'
        
        print(f"Processing: {filename}")
        
        if not os.path.exists(backup_path):
            print(f"  ✗ No backup file found: {backup_path}\n")
            failed_count += 1
            continue
        
        try:
            if plot_comparison(csv_path, backup_path, output_dir):
                success_count += 1
                print(f"  ✓ Successfully created plot for {filename}\n")
            else:
                failed_count += 1
                print(f"  ✗ Failed to create plot for {filename}\n")
        except Exception as e:
            failed_count += 1
            print(f"  ✗ Error processing {filename}: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("=" * 50)
    print(f"Plotting complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
