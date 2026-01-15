"""
Complementary Filter for Orientation Estimation

Python implementation of the C++ Estimator filter for orientation estimation.
This module recreates the complementary filter that estimates YPR (Yaw-Pitch-Roll)
and their velocities from IMU measurements (gyroscopes and accelerometers) and
motor encoder data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from wheelbot_dataset import Dataset
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


def run_filter_on_experiment(exp, start_time=None, end_time=None):
    """
    Run the complementary filter on an experiment.
    
    Args:
        exp: Experiment object
        start_time: Optional start time for filtering (seconds)
        end_time: Optional end time for filtering (seconds)
        
    Returns:
        Dictionary with time, estimated orientation, and velocities
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
    
    return {
        'time': time,
        'yaw': yaw,
        'roll': roll,
        'pitch': pitch,
        'yaw_vel': yaw_vel,
        'roll_vel': roll_vel,
        'pitch_vel': pitch_vel
    }


def plot_filter_results(time, filter_results, reference_data=None, vicon_data=None, 
                       onboard_data=None, title="Filter Results"):
    """
    Plot the filter estimation results with optional ground truth comparison.
    
    Args:
        time: Time array
        filter_results: Dictionary with filter outputs (from complementary filter)
        reference_data: Optional dictionary with reference data (generic reference)
        vicon_data: Optional dictionary with Vicon ground truth data {'yaw', 'roll', 'pitch'}
        onboard_data: Optional dictionary with onboard estimate data {'yaw', 'roll', 'pitch'}
        title: Plot title
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    
    angles = ['roll', 'pitch', 'yaw']  # Reorder to match standard RPY convention
    labels = ['Roll', 'Pitch', 'Yaw']
    
    for i, (angle, label) in enumerate(zip(angles, labels)):
        ax = axes[i]
        
        # Plot filter estimate (primary result)
        ax.plot(time, filter_results[angle], 
               label=f'{label} (Complementary Filter)', 
               linewidth=1.5, color='blue')
        
        # Plot onboard estimate if provided
        if onboard_data and angle in onboard_data:
            ax.plot(time, onboard_data[angle], 
                   label=f'{label} (Onboard Estimate)', 
                   linestyle='--', linewidth=1.0, color='orange', alpha=0.8)
        
        # Plot Vicon ground truth if provided
        if vicon_data and angle in vicon_data:
            vicon_angles = vicon_data[angle].copy()
            
            # Align yaw offset (Vicon and filter may have different initial yaw)
            if angle == 'yaw':
                vicon_angles = vicon_angles - np.mean(vicon_angles) + np.mean(filter_results[angle])
            
            ax.plot(time, vicon_angles, 
                   label=f'{label} (Vicon Ground Truth)', 
                   linestyle=':', linewidth=1.5, color='green', alpha=0.8)
        
        # Plot generic reference if provided (and no specific onboard/vicon data)
        if reference_data and angle in reference_data and not onboard_data and not vicon_data:
            ax.plot(time, reference_data[angle], 
                   label=f'{label} (Reference)', 
                   linestyle='--', alpha=0.7)
        
        ax.set_ylabel(f'{label} [rad]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} Angle')
    
    plt.tight_layout()
    return fig


def main():
    """
    Main function to run the complementary filter example.
    
    Processes experiments from the dataset and generates comparison plots
    with onboard estimates and Vicon ground truth (when available).
    """
    # Load dataset
    ds = Dataset("../../data")
    
    # Select groups to process
    groups_to_process = ["yaw", "pitch", "roll"]
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    print("=" * 80)
    print("COMPLEMENTARY FILTER ESTIMATION EXAMPLE")
    print("=" * 80)
    print()
    
    for group_name in groups_to_process:
        if group_name not in ds.groups:
            print(f"Group '{group_name}' not found in dataset, skipping...")
            continue
        
        group = ds.groups[group_name]
        print(f"\nProcessing group: {group_name}")
        print(f"  Number of experiments: {len(group.experiments)}")
        
        # Process first 3 experiments as examples
        with PdfPages(f"plots/filter_estimation_{group_name}.pdf") as pdf:
            for exp_idx, exp in enumerate(group.experiments[:3]):
                print(f"  Processing experiment {exp_idx}...")
                
                # Run filter
                filter_results = run_filter_on_experiment(exp)
                time = filter_results['time']
                
                # Get onboard estimate data if available
                onboard_data = None
                if all(f'/q_yrp/{a}' in exp.data.columns for a in ['yaw', 'roll', 'pitch']):
                    # Slice to match filter results time range
                    df = exp.data
                    if df.index.name == '_time':
                        mask = (df.index >= time[0]) & (df.index <= time[-1])
                    else:
                        mask = (df['_time'] >= time[0]) & (df['_time'] <= time[-1])
                    
                    df_sliced = df[mask]
                    onboard_data = {
                        'yaw': df_sliced['/q_yrp/yaw'].to_numpy(),
                        'roll': df_sliced['/q_yrp/roll'].to_numpy(),
                        'pitch': df_sliced['/q_yrp/pitch'].to_numpy()
                    }
                
                # Get Vicon ground truth data if available
                vicon_data = None
                vicon_quat_fields = [
                    "/vicon_orientation_wxyz/w",
                    "/vicon_orientation_wxyz/x",
                    "/vicon_orientation_wxyz/y",
                    "/vicon_orientation_wxyz/z"
                ]
                
                if all(f in exp.data.columns for f in vicon_quat_fields):
                    df = exp.data
                    if df.index.name == '_time':
                        mask = (df.index >= time[0]) & (df.index <= time[-1])
                    else:
                        mask = (df['_time'] >= time[0]) & (df['_time'] <= time[-1])
                    
                    df_sliced = df[mask]
                    
                    qw = df_sliced["/vicon_orientation_wxyz/w"].to_numpy()
                    qx = df_sliced["/vicon_orientation_wxyz/x"].to_numpy()
                    qy = df_sliced["/vicon_orientation_wxyz/y"].to_numpy()
                    qz = df_sliced["/vicon_orientation_wxyz/z"].to_numpy()
                    
                    # Check if vicon data is valid (not all zeros)
                    if not (np.all(qw == 0) and np.all(qx == 0) and np.all(qy == 0) and np.all(qz == 0)):
                        yaw_vicon, roll_vicon, pitch_vicon = quaternion_to_euler_zxy(qw, qx, qy, qz)
                        vicon_data = {
                            'yaw': yaw_vicon,
                            'roll': roll_vicon,
                            'pitch': pitch_vicon
                        }
                        print(f"    ✓ Vicon ground truth data available")
                    else:
                        print(f"    ⚠ Vicon data is all zeros (no ground truth)")
                else:
                    print(f"    ⚠ No Vicon data available")
                
                # Plot results
                title_parts = [f"Filter Estimation: {group_name} - Experiment {exp_idx}"]
                if vicon_data:
                    title_parts.append("(with Vicon Ground Truth)")
                
                fig = plot_filter_results(
                    time,
                    filter_results,
                    vicon_data=vicon_data,
                    onboard_data=onboard_data,
                    title=" ".join(title_parts)
                )
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"  Plots saved to: plots/filter_estimation_{group_name}.pdf")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
