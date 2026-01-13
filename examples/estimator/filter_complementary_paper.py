"""
Python implementation of the C++ Estimator filter for orientation estimation.

This script recreates the complementary filter that estimates YPR (Yaw-Pitch-Roll)
and their velocities from IMU measurements (gyroscopes and accelerometers) and
motor encoder data.
"""

import sys
import os

# Add parent directory to path to import dataset module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dataset import Dataset
from matplotlib.backends.backend_pdf import PdfPages


def quaternion_to_euler_zxy(w, x, y, z):
    """
    Convert quaternion to Euler angles using ZXY convention.
    
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


class Estimator:
    """
    Python implementation of the C++ Estimator class.
    
    Estimates orientation (YRP) and angular velocities using a complementary
    filter that fuses gyroscope and accelerometer measurements.
    """
    
    def __init__(self, N_IMUS=4, N_MOTORS=2):
        self.N_IMUS = N_IMUS
        self.N_MOTORS = N_MOTORS
        
        # Rotation matrices
        self.R_upside_down = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ], dtype=np.float32)
        
        self.R01 = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ], dtype=np.float32)
        
        self.R23 = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.R_Bi = [self.R01, self.R01, self.R23, self.R23]
        
        # IMU positions
        self.X1 = np.array([-0.166896, -0.167463, 0.667201, 0.667159], dtype=np.float32)
        
        # State variables
        self.upside_down = False
        self.pivot_accel = np.zeros(3, dtype=np.float32)
        
        # Gyro estimations
        self.q_G = np.zeros(3, dtype=np.float32)
        self.dq_G = np.zeros(3, dtype=np.float32)
        
        # Accel estimations
        self.q_A = np.zeros(2, dtype=np.float32)
        
        # Euler angles and derivatives
        self.q = np.zeros(3, dtype=np.float32)
        self.dq = np.zeros(3, dtype=np.float32)
        self.ddq = np.zeros(3, dtype=np.float32)
        
        # Encoder
        self.q_WR = np.zeros(N_MOTORS, dtype=np.float32)
        self.dq_WR = np.zeros(N_MOTORS, dtype=np.float32)
        self.ddq_WR = np.zeros(N_MOTORS, dtype=np.float32)
        
        # Parameters
        self.dt = 1e-3  # 1ms timestep
        self.r = 32e-3  # Wheel radius in meters
        self.alpha = 0.01  # Complementary filter fusion parameter
        
        self.init = True
        self.use_pivot_accel = True
    
    def R1(self, q1):
        """Rotation matrix around x-axis."""
        c = np.cos(q1)
        s = np.sin(q1)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=np.float32)
    
    def R2(self, q2):
        """Rotation matrix around y-axis."""
        c = np.cos(q2)
        s = np.sin(q2)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float32)
    
    def R3(self, q3):
        """Rotation matrix around z-axis."""
        c = np.cos(q3)
        s = np.sin(q3)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def jacobian_w2euler(self, q1, q2):
        """Jacobian matrix for converting body rates to Euler rates."""
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        t1 = np.tan(q1)
        
        J = np.array([
            [c2, 0, s2],
            [s2 * t1, 1, -c2 * t1],
            [-s2 / c1, 0, c2 / c1]
        ], dtype=np.float32)
        
        return J
    
    def average_vecs(self, m):
        """Average vectors across IMUs."""
        return np.mean(m, axis=1)
    
    def estimate_gyro(self, w_B, q):
        """Estimate Euler angle rates from body rates."""
        J = self.jacobian_w2euler(q[0], q[1])
        w_avg = self.average_vecs(w_B)
        return J @ w_avg
    
    def integrate(self, dq, past_q, past_dq, dt):
        """Trapezoidal integration."""
        return (dq + past_dq) / 2.0 * dt + past_q
    
    def calculate_g_B(self, m_B, pivot_acc, X1):
        """Calculate gravity vector in body frame."""
        M = m_B - pivot_acc.reshape(-1, 1)
        return M @ X1
    
    def estimate_accel(self, g_B):
        """Estimate roll and pitch from gravity vector."""
        q_A = np.zeros(2, dtype=np.float32)
        # q_A[0] = np.arctan(g_B[1]/np.sqrt(g_B[0]**2 + g_B[2]**2))
        # q_A[1] = -np.arctan(g_B[0]/g_B[2])
        q_A[0] = np.arctan2(g_B[1], np.sqrt(g_B[0]**2 + g_B[2]**2))
        q_A[1] = np.arctan2(-g_B[0], g_B[2])
        return q_A
    
    def estimate_pivot_accel(self, q, dq, ddq, dq4, ddq4):
        """Estimate pivot point acceleration."""
        c1 = np.cos(q[0])
        s1 = np.sin(q[0])
        
        temp1 = np.array([
            2 * c1 * self.r * dq[0] * dq[2] + self.r * s1 * ddq[2],
            -c1 * self.r * ddq[0] + self.r * s1 * dq[0]**2 + self.r * s1 * dq[2]**2,
            -c1 * self.r * dq[0]**2 - self.r * s1 * ddq[0]
        ], dtype=np.float32)
        
        ddp_WC = self.R2(q[1]).T @ self.R1(q[0]).T @ temp1
        
        temp2 = np.array([
            self.r * (ddq4 + ddq[1]),
            self.r * dq[2] * (dq4 + dq[1]),
            0
        ], dtype=np.float32)
        
        ddp_CI = self.R2(q[1]).T @ self.R1(q[0]).T @ temp2
        
        return ddp_WC + ddp_CI
    
    def IIR_filter(self, new_val, old_val, alpha):
        """Simple IIR low-pass filter."""
        return alpha * new_val + (1 - alpha) * old_val
    
    def update(self, omega_B, a_B, motor_states):
        """
        Main update step of the estimator.
        
        Args:
            omega_B: 3xN_IMUS matrix of gyro measurements (rad/s)
            a_B: 3xN_IMUS matrix of accelerometer measurements (m/s^2)
            motor_states: 3xN_MOTORS matrix of [position, velocity, acceleration]
        
        Returns:
            10-element state vector: [yaw, roll, pitch, yaw_vel, roll_vel, pitch_vel,
                                     q_wheel, dq_wheel, q_reaction, dq_reaction]
        """
        if self.init:
            g_B = self.calculate_g_B(a_B, self.pivot_accel, self.X1)
            self.q_A = self.estimate_accel(g_B)
            self.q[0:2] = self.q_A
            self.q[2] = 0
            self.init = False
        
        # Extract motor states
        m_q_WR = motor_states[0, :]
        m_dq_WR = motor_states[1, :]
        m_ddq_WR = motor_states[2, :]
        
        self.ddq_WR = m_ddq_WR
        self.dq_WR = m_dq_WR
        self.q_WR = m_q_WR
        
        # Calculate body rates from gyro measurements and integrate
        self.dq_G = self.estimate_gyro(omega_B, self.q)
        self.q_G = self.integrate(self.dq_G, self.q, self.dq_G, self.dt)
        
        # Second derivative by numerically differentiating gyro angular rate and IIR filtering
        self.ddq = self.IIR_filter((self.dq_G - self.dq) / self.dt, self.ddq, 0.1)
        
        # Accel estimation
        if self.use_pivot_accel:
            self.pivot_accel = self.estimate_pivot_accel(
                self.q, self.dq_G, self.ddq, self.dq_WR[0], self.ddq_WR[0]
            )
        else:
            self.pivot_accel = np.zeros(3, dtype=np.float32)
        
        g_B = self.calculate_g_B(a_B, self.pivot_accel, self.X1)
        self.q_A = self.estimate_accel(g_B)
        
        # Complementary filter
        alpha_scale = self.alpha
        self.q[0] = alpha_scale * self.q_A[0] + (1 - alpha_scale) * self.q_G[0]
        self.q[1] = alpha_scale * self.q_A[1] + (1 - alpha_scale) * self.q_G[1]
        self.q[2] = self.q_G[2]
        
        self.dq = self.dq_G
        
        # Return state vector: [yaw, roll, pitch, yaw_vel, roll_vel, pitch_vel, q_wheel, dq_wheel, q_reaction, dq_reaction]
        return np.array([
            self.q[2], self.q[0], self.q[1],
            self.dq[2], self.dq[0], self.dq[1],
            self.q_WR[0], self.dq_WR[0], self.q_WR[1], self.dq_WR[1]
        ], dtype=np.float32)


def run_filter_on_experiment(exp):
    """
    Run the estimator filter on a single experiment.
    
    Args:
        exp: Experiment object with raw data
        
    Returns:
        Dictionary with time and estimated states
    """
    df = exp.data
    
    # Check if all required fields are present
    gyro_fields = [f"/gyro{i}/{axis}" for i in range(4) for axis in ['x', 'y', 'z']]
    accel_fields = [f"/accel{i}/{axis}" for i in range(4) for axis in ['x', 'y', 'z']]
    motor_fields = ['/q_DR/drive_wheel', '/dq_DR/drive_wheel', '/ddq_DR/drive_wheel']
    
    required_fields = gyro_fields + accel_fields + motor_fields
    missing_fields = [f for f in required_fields if f not in df.columns]
    
    if missing_fields:
        print(f"    Missing fields: {missing_fields[:5]}...")
        return None
    
    # Initialize estimator
    estimator = Estimator(N_IMUS=4, N_MOTORS=2)
    
    # Prepare data storage
    n_samples = len(df)
    time = df.index.to_numpy()
    
    # Results storage
    yaw = np.zeros(n_samples)
    roll = np.zeros(n_samples)
    pitch = np.zeros(n_samples)
    yaw_vel = np.zeros(n_samples)
    roll_vel = np.zeros(n_samples)
    pitch_vel = np.zeros(n_samples)
    
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
        # Reaction wheel is index 1, but we'll use drive wheel data for both if reaction not available
        if '/q_DR/reaction_wheel' in df.columns:
            motor_states[0, 1] = df['/q_DR/reaction_wheel'].iloc[i]
            motor_states[1, 1] = df['/dq_DR/reaction_wheel'].iloc[i]
            motor_states[2, 1] = df['/ddq_DR/reaction_wheel'].iloc[i]
        else:
            motor_states[:, 1] = motor_states[:, 0]
        
        # Update estimator
        state = estimator.update(omega_B, a_B, motor_states)
        
        # Store results
        yaw[i] = state[0]
        roll[i] = state[1]
        pitch[i] = state[2]
        yaw_vel[i] = state[3]
        roll_vel[i] = state[4]
        pitch_vel[i] = state[5]
    
    return {
        'time': time,
        'yaw': yaw,
        'roll': roll,
        'pitch': pitch,
        'yaw_vel': yaw_vel,
        'roll_vel': roll_vel,
        'pitch_vel': pitch_vel
    }


def main():
    # Load the dataset from the data folder (raw data, no preprocessing)
    ds = Dataset("../../data")
    
    # Create output directory if it doesn't exist
    os.makedirs("plots/estimator", exist_ok=True)
    
    # Process each group and experiment
    for group_name, group in ds.groups.items():
        print(f"\nProcessing group: {group_name}")
        
        for exp_idx, exp in enumerate(group.experiments):
            print(f"  Experiment {exp_idx}")
            
            # Run the filter
            result = run_filter_on_experiment(exp)
            
            if result is None:
                print(f"    Skipping due to missing fields")
                continue
            
            time_full = result['time']
            
            # Cut data from 20s to 40s
            mask_window = (time_full >= 20.0) & (time_full <= 40.0)
            if not np.any(mask_window):
                print(f"    Skipping: no data in 20-40s range")
                continue
            
            indices = np.where(mask_window)[0]
            start_idx = indices[0]
            end_idx = indices[-1]
            
            if start_idx >= end_idx:
                print(f"    Skipping: insufficient data after cutting")
                continue
            
            # Cut the data for visualization
            time_cut = time_full[start_idx:end_idx+1]
            roll_cut = result['roll'][start_idx:end_idx+1]
            pitch_cut = result['pitch'][start_idx:end_idx+1]
            yaw_cut = result['yaw'][start_idx:end_idx+1]
            
            # Extract onboard q_yrp estimates if available
            has_q_yrp = False # all(f in exp.data.columns for f in ['/q_yrp/yaw', '/q_yrp/roll', '/q_yrp/pitch'])
            if has_q_yrp:
                yaw_onboard = exp.data['/q_yrp/yaw'].to_numpy()[start_idx:end_idx+1]
                roll_onboard = exp.data['/q_yrp/roll'].to_numpy()[start_idx:end_idx+1]
                pitch_onboard = exp.data['/q_yrp/pitch'].to_numpy()[start_idx:end_idx+1]
            
            # Extract Vicon data if available
            vicon_quat_fields = [
                "/vicon_orientation_wxyz/w",
                "/vicon_orientation_wxyz/x",
                "/vicon_orientation_wxyz/y",
                "/vicon_orientation_wxyz/z"
            ]
            
            has_vicon = all(f in exp.data.columns for f in vicon_quat_fields)
            
            if has_vicon:
                qw = exp.data["/vicon_orientation_wxyz/w"].to_numpy()
                qx = exp.data["/vicon_orientation_wxyz/x"].to_numpy()
                qy = exp.data["/vicon_orientation_wxyz/y"].to_numpy()
                qz = exp.data["/vicon_orientation_wxyz/z"].to_numpy()
                
                yaw_vicon, roll_vicon, pitch_vicon = quaternion_to_euler_zxy(qw, qx, qy, qz)
                
                # Cut vicon data
                roll_vicon_cut = roll_vicon[start_idx:end_idx+1]
                pitch_vicon_cut = pitch_vicon[start_idx:end_idx+1]
                yaw_vicon_cut = yaw_vicon[start_idx:end_idx+1]
                
                # Align vicon roll and yaw to match filter estimate (average over entire sequence)
                roll_vicon_cut = roll_vicon_cut - np.mean(roll_vicon_cut) + np.mean(roll_cut)
                yaw_vicon_cut = yaw_vicon_cut - np.mean(yaw_vicon_cut) + np.mean(yaw_cut)
            
            # Create plots
            # Use original dimensions as requested
            width = 3.4125
            height = 0.8
            fig, axes = plt.subplots(1, 3, figsize=(width, height))
            
            # Style settings from user example
            rc('font', size=8)
            rc('text', usetex=False)
            title_fontsize = 6
            tick_fontsize = 5
            linewidth = 0.8
            
            # Roll
            axes[0].plot(time_cut, roll_cut, label='Estimator', linewidth=linewidth)
            if has_vicon:
                axes[0].plot(time_cut, roll_vicon_cut, label='Vicon (GT)', linewidth=linewidth, linestyle='--', alpha=0.7)
            if has_q_yrp:
                axes[0].plot(time_cut, roll_onboard, label='Onboard', linewidth=linewidth, linestyle=':', alpha=0.8)
            
            axes[0].set_ylabel('Angle [rad]', fontsize=tick_fontsize, labelpad=1)
            # axes[0].set_title('Roll', fontsize=title_fontsize, pad=1)
            axes[0].text(0.8, 0.9, 'roll', transform=axes[0].transAxes, fontsize=title_fontsize, ha='center', va='top', bbox=dict(facecolor='#FFDFC3', alpha=1, edgecolor='none', pad=1))
            axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=0.5)
            axes[0].grid(True, linewidth=0.3, alpha=0.6, linestyle='--')
            
            # Pitch
            axes[1].plot(time_cut, pitch_cut, label='Estimator', linewidth=linewidth)
            if has_vicon:
                axes[1].plot(time_cut, pitch_vicon_cut, label='Vicon', linewidth=linewidth, linestyle='--', alpha=0.7)
            if has_q_yrp:
                axes[1].plot(time_cut, pitch_onboard, label='Onboard', linewidth=linewidth, linestyle=':', alpha=0.8)
            
            # axes[1].set_title('Pitch', fontsize=title_fontsize, pad=1)
            axes[1].text(0.5, 0.9, 'pitch', transform=axes[1].transAxes, fontsize=title_fontsize, ha='center', va='top', bbox=dict(facecolor='#E5F3E5', alpha=1, edgecolor='none', pad=1))
            axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=0.5)
            axes[1].grid(True, linewidth=0.3, alpha=0.6, linestyle='--')
            
            # Yaw
            axes[2].plot(time_cut, yaw_cut, label='Estimator', linewidth=linewidth)
            if has_vicon:
                axes[2].plot(time_cut, yaw_vicon_cut, label='Vicon', linewidth=linewidth, linestyle='--', alpha=0.7)
            if has_q_yrp:
                axes[2].plot(time_cut, yaw_onboard, label='Onboard', linewidth=linewidth, linestyle=':', alpha=0.8)
            
            # axes[2].set_title('Yaw', fontsize=title_fontsize, pad=1)
            axes[2].text(0.2, 0.9, 'yaw', transform=axes[2].transAxes, fontsize=title_fontsize, ha='center', va='top', bbox=dict(facecolor='#E3EEF5', alpha=1, edgecolor='none', pad=1))
            axes[2].tick_params(axis='both', which='major', labelsize=tick_fontsize, pad=0.5)
            axes[2].grid(True, linewidth=0.3, alpha=0.6, linestyle='--')
            
            # Legend in bottom right of Yaw plot
            handles, labels = axes[0].get_legend_handles_labels()
            axes[2].legend(handles, labels, loc='lower right', fontsize=5, frameon=True, labelspacing=0.1, handlelength=1.0)
            
            for ax in axes:
                ax.set_xlabel('Time [s]', fontsize=tick_fontsize, labelpad=1)

            plt.tight_layout(pad=0.2)
            
            # Save the figure
            output_path = f"plots/estimator/filter_{group_name}_exp{exp_idx}_paper.pdf"
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            
            print(f"    Saved: {output_path}")
    
    print("\nFilter estimation complete!")


if __name__ == "__main__":
    main()
