"""
Orientation comparison between onboard estimator and Vicon ground truth.

This script loads experimental data, extracts IMU sensor readings (gyros),
onboard orientation estimates, and Vicon ground truth orientation. It then
converts the Vicon quaternions to Euler angles and compares them with the
onboard estimates in separate subplots for roll, pitch, and yaw.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from wheelbot_dataset import Dataset
from matplotlib.backends.backend_pdf import PdfPages


def quaternion_to_euler_zxy(w, x, y, z):
    """
    Convert quaternion to Euler angles using ZXY convention.
    
    The convention is: first rotate by yaw around Z, then roll around X, then pitch around Y.
    This matches the yaw-roll-pitch convention specified in the requirements.
    
    Args:
        w, x, y, z: Quaternion components (scalar-first convention)
    
    Returns:
        yaw, roll, pitch: Euler angles in radians
    """
    # Roll (rotation around x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (rotation around y-axis)
    sinp = 2 * (w * y - z * x)
    # Handle singularity
    pitch = np.where(
        np.abs(sinp) >= 1,
        np.copysign(np.pi / 2, sinp),
        np.arcsin(sinp)
    )
    
    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return yaw, roll, pitch


def main():
    # Load the dataset from the data folder
    ds = Dataset("../../data")
    
    # Apply simple preprocessing: cut zero actions and resample
    cut_and_resample_fn = lambda exp: (
        exp
        .cut_by_condition(
            start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
            end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0,
        )
        .cut_time(start=2.0, end=2.0)
        .resample(dt=0.01)  # 10ms = 0.01s
    )
    
    # Apply to entire dataset
    ds_processed = ds.map(cut_and_resample_fn)
    
    # Fields to extract
    gyro_fields = [
        "/gyro0/x", "/gyro0/y", "/gyro0/z",
        "/gyro1/x", "/gyro1/y", "/gyro1/z",
        "/gyro2/x", "/gyro2/y", "/gyro2/z",
        "/gyro3/x", "/gyro3/y", "/gyro3/z"
    ]
    
    yrp_fields = [
        "/q_yrp/yaw", "/q_yrp/roll", "/q_yrp/pitch"
    ]
    
    yrp_vel_fields = [
        "/dq_yrp/yaw_vel", "/dq_yrp/roll_vel", "/dq_yrp/pitch_vel"
    ]
    
    vicon_quat_fields = [
        "/vicon_orientation_wxyz/w",
        "/vicon_orientation_wxyz/x",
        "/vicon_orientation_wxyz/y",
        "/vicon_orientation_wxyz/z"
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs("plots/estimator", exist_ok=True)
    
    # Process each group and experiment
    for group_name, group in ds_processed.groups.items():
        print(f"\nProcessing group: {group_name}")
        
        for exp_idx, exp in enumerate(group.experiments):
            print(f"  Experiment {exp_idx}")
            
            # Check if all required fields are present
            required_fields = yrp_fields + vicon_quat_fields
            missing_fields = [f for f in required_fields if f not in exp.data.columns]
            
            if missing_fields:
                print(f"    Skipping: missing fields {missing_fields}")
                continue
            
            # Extract data
            df = exp.data
            time = df.index.to_numpy()
            
            # Onboard estimates
            yaw_onboard = df["/q_yrp/yaw"].to_numpy()
            roll_onboard = df["/q_yrp/roll"].to_numpy()
            pitch_onboard = df["/q_yrp/pitch"].to_numpy() * 2  # Multiply pitch by 2
            
            # Vicon quaternion
            qw = df["/vicon_orientation_wxyz/w"].to_numpy()
            qx = df["/vicon_orientation_wxyz/x"].to_numpy()
            qy = df["/vicon_orientation_wxyz/y"].to_numpy()
            qz = df["/vicon_orientation_wxyz/z"].to_numpy()
            
            # Convert Vicon quaternion to Euler angles
            yaw_vicon, roll_vicon, pitch_vicon = quaternion_to_euler_zxy(qw, qx, qy, qz)
            
            # Adjust Vicon roll angle to match onboard estimate at first value
            roll_vicon = roll_vicon - roll_vicon[0] + roll_onboard[0]
            
            # Reset other Vicon values to start at 0 (subtract first value)
            pitch_vicon = pitch_vicon - pitch_vicon[0]
            yaw_vicon = yaw_vicon - yaw_vicon[0]
            
            # Onboard angular velocities
            roll_vel_onboard = df["/dq_yrp/roll_vel"].to_numpy()
            pitch_vel_onboard = df["/dq_yrp/pitch_vel"].to_numpy()
            yaw_vel_onboard = df["/dq_yrp/yaw_vel"].to_numpy()
            
            # Compute angular velocities from Vicon using central difference
            dt = time[1] - time[0]  # Assumes uniform sampling
            roll_vel_vicon = np.gradient(roll_vicon, dt)
            pitch_vel_vicon = np.gradient(pitch_vicon, dt)
            yaw_vel_vicon = np.gradient(yaw_vicon, dt)
            
            # Convert degree limits to radians
            roll_lim = np.deg2rad(15)
            pitch_lim = np.deg2rad(30)
            yaw_lim = np.deg2rad(45)
            
            # Create comparison plots
            fig, axes = plt.subplots(6, 1, figsize=(12, 18))
            fig.suptitle(f"Orientation Comparison: {group_name} - Experiment {exp_idx}", fontsize=14)
            
            # Roll comparison
            axes[0].plot(time, roll_onboard, label='Onboard Estimate', linewidth=1.5)
            axes[0].plot(time, roll_vicon, label='Vicon Ground Truth', linewidth=1.5, linestyle='--')
            axes[0].set_ylabel('Roll [rad]')
            axes[0].set_xlabel('Time [s]')
            axes[0].set_ylim(-roll_lim, roll_lim)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Roll Angle')
            
            # Pitch comparison
            axes[1].plot(time, pitch_onboard, label='Onboard Estimate', linewidth=1.5)
            axes[1].plot(time, pitch_vicon, label='Vicon Ground Truth', linewidth=1.5, linestyle='--')
            axes[1].set_ylabel('Pitch [rad]')
            axes[1].set_xlabel('Time [s]')
            axes[1].set_ylim(-pitch_lim, pitch_lim)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Pitch Angle')
            
            # Yaw comparison
            axes[2].plot(time, yaw_onboard, label='Onboard Estimate', linewidth=1.5)
            axes[2].plot(time, yaw_vicon, label='Vicon Ground Truth', linewidth=1.5, linestyle='--')
            axes[2].set_ylabel('Yaw [rad]')
            axes[2].set_xlabel('Time [s]')
            axes[2].set_ylim(-yaw_lim, yaw_lim)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title('Yaw Angle')
            
            # Roll velocity comparison
            axes[3].plot(time, roll_vel_onboard, label='Onboard Estimate', linewidth=1.5)
            axes[3].plot(time, roll_vel_vicon, label='Vicon (Central Diff)', linewidth=1.5, linestyle='--')
            axes[3].set_ylabel('Roll Velocity [rad/s]')
            axes[3].set_xlabel('Time [s]')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            axes[3].set_title('Roll Angular Velocity')
            
            # Pitch velocity comparison
            axes[4].plot(time, pitch_vel_onboard, label='Onboard Estimate', linewidth=1.5)
            axes[4].plot(time, pitch_vel_vicon, label='Vicon (Central Diff)', linewidth=1.5, linestyle='--')
            axes[4].set_ylabel('Pitch Velocity [rad/s]')
            axes[4].set_xlabel('Time [s]')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
            axes[4].set_title('Pitch Angular Velocity')
            
            # Yaw velocity comparison
            axes[5].plot(time, yaw_vel_onboard, label='Onboard Estimate', linewidth=1.5)
            axes[5].plot(time, yaw_vel_vicon, label='Vicon (Central Diff)', linewidth=1.5, linestyle='--')
            axes[5].set_ylabel('Yaw Velocity [rad/s]')
            axes[5].set_xlabel('Time [s]')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
            axes[5].set_title('Yaw Angular Velocity')
            
            plt.tight_layout()
            
            # Save the figure
            output_path = f"plots/estimator/orientation_{group_name}_exp{exp_idx}.pdf"
            plt.savefig(output_path)
            plt.close()
            
            print(f"    Saved: {output_path}")
    
    print("\nOrientation comparison complete!")


if __name__ == "__main__":
    main()
