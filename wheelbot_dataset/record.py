#!/usr/bin/env python3
"""
Main entry point for recording wheelbot experiments.

This script provides various experiment recording functions that can be
invoked from the command line using the fire CLI.

Usage:
    python -m wheelbot_dataset.record vel
    python -m wheelbot_dataset.record roll
    python -m wheelbot_dataset.record pitch
    python -m wheelbot_dataset.record velrollpitch
    python -m wheelbot_dataset.record consolidate archive data_consolidated
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
import re
import shutil

from wheelbot_dataset.recording.experiment import plot_and_run_with_repeat
from wheelbot_dataset.recording.prb_sequences import (
    generate_setpoints, convert_yaw_setpoints_to_deltas, generate_yaw_prbs
)
from wheelbot_dataset.recording.utils import (
    next_log_number, continue_skip_abort
)

# Note: trivial_sequences module functions need to be defined or imported
# These functions were referenced in the original record.py but the module doesn't exist
# Placeholder implementations are provided below

import fire

default_wheelbot_name="wheelbot-beta-2"
# default_surface="gray_felt"
default_surface="black_pvc"
default_video_device="/dev/video4"
# default_video_device=None
global_seed_offset = 0


# Placeholder for trivial_sequences functions
def linear_sequence(target_velocity, accel_time=1.0, const_time=1.0, dt=0.05):
    """Generate a linear velocity sequence."""
    accel_samples = int(accel_time / dt)
    const_samples = int(const_time / dt)
    decel_samples = int(accel_time / dt)
    
    velocity = np.concatenate([
        np.linspace(0, target_velocity, accel_samples),
        np.ones(const_samples) * target_velocity,
        np.linspace(target_velocity, 0, decel_samples)
    ])
    
    roll = np.zeros_like(velocity)
    pitch = np.zeros_like(velocity)
    time = np.arange(len(velocity)) * dt
    
    return velocity, roll, pitch, time, dt


def linear_velocity_sequence(target_velocity=1.0, const_time=1.0, accel_time=1.0, dt=0.05):
    """Generate a linear velocity sequence returning velocity, time, dt."""
    accel_samples = int(accel_time / dt)
    const_samples = int(const_time / dt)
    decel_samples = int(accel_time / dt)
    
    velocity = np.concatenate([
        np.linspace(0, target_velocity, accel_samples),
        np.ones(const_samples) * target_velocity,
        np.linspace(target_velocity, 0, decel_samples)
    ])
    
    time = np.arange(len(velocity)) * dt
    
    return velocity, time, dt


def linear_sequence_with_lean(target_velocity=1.0, const_time=1.0, accel_time=1.0, 
                               pitch_accel_deg=5.0, pitch_decel_deg=-5.0, dt=0.05):
    """Generate a linear velocity sequence with pitch lean during acceleration/deceleration."""
    accel_samples = int(accel_time / dt)
    const_samples = int(const_time / dt)
    decel_samples = int(accel_time / dt)
    
    velocity = np.concatenate([
        np.linspace(0, target_velocity, accel_samples),
        np.ones(const_samples) * target_velocity,
        np.linspace(target_velocity, 0, decel_samples)
    ])
    
    roll = np.zeros_like(velocity)
    pitch = np.concatenate([
        np.ones(accel_samples) * pitch_accel_deg,
        np.zeros(const_samples),
        np.ones(decel_samples) * pitch_decel_deg
    ])
    
    time = np.arange(len(velocity)) * dt
    
    return velocity, roll, pitch, time, dt


def build_linear_velocity_profile_for_other(other_signal, dt, target_velocity=0.5):
    """Build a velocity profile that ramps up and down around another signal."""
    n = len(other_signal)
    ramp_samples = int(1.0 / dt)  # 1 second ramp
    
    velocity = np.ones(n) * target_velocity
    velocity[:ramp_samples] = np.linspace(0, target_velocity, ramp_samples)
    velocity[-ramp_samples:] = np.linspace(target_velocity, 0, ramp_samples)
    
    return velocity


def vel(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    seed = 0
    vel_ranges = np.linspace(0.1,1,5)
    for seed, vr in enumerate(vel_ranges):
        print(f"Velocity: {vr}")
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/velocity")
        
        velocity, roll, pitch = generate_setpoints(
            duration_s=(duration:=10),
            stabilize_every_n_seconds = 10,
            stabilize_for = 5,
            dt=(dt:=0.05),
            random_seed=seed + global_seed_offset,
            roll_range_deg=0,
            pitch_range_deg=0,
            vel_range=vr,
            vel_slew_per_0p2s=0.5,
            roll_pert_stddev=0,
            pitch_pert_stddev=0,
            vel_pert_stddev=0
        )
        time = np.arange(0, duration, dt)

        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)            

def roll(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    roll_ranges = np.linspace(4,10,5)
    for seed, rr in enumerate(roll_ranges):
        print(f"Roll: {rr}")
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/roll")
        stabilize_interval = 3 if rr >= 8 else 5
        velocity, roll, pitch = generate_setpoints(
            duration_s=(duration:=30.0),
            stabilize_every_n_seconds=stabilize_interval,
            stabilize_for=3,
            dt=(dt:=0.05),
            random_seed=seed + global_seed_offset,
            roll_range_deg=rr,
            pitch_range_deg=0,
            vel_range=0,
            vel_slew_per_0p2s=1,
            roll_pert_stddev=0,
            pitch_pert_stddev=0,
            vel_pert_stddev=0
        )
        time = np.arange(0, duration, dt)
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)

def roll_max(angle=10, wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    for seed in range(10):
        print(f"Roll max: {angle} (seed: {seed})")
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/roll_max")
        stabilize_interval = 3 if angle >= 8 else 5
        velocity, roll, pitch = generate_setpoints(
            duration_s=(duration:=30.0),
            stabilize_every_n_seconds=stabilize_interval,
            stabilize_for=3,
            dt=(dt:=0.05),
            random_seed=seed + global_seed_offset,
            roll_range_deg=angle,
            pitch_range_deg=0,
            vel_range=0,
            vel_slew_per_0p2s=1,
            roll_pert_stddev=0,
            pitch_pert_stddev=0,
            vel_pert_stddev=0
        )
        time = np.arange(0, duration, dt)
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)

def pitch(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    pitch_ranges = np.linspace(15,20,3)
    for seed, pr in enumerate(pitch_ranges):
        print(f"Pitch: {pr}")
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/pitch")
        stabilize_interval = 3 if pr >= 4 else 5
        velocity, roll, pitch = generate_setpoints(
            duration_s=(duration:=30.0),
            stabilize_every_n_seconds=stabilize_interval,
            stabilize_for=3,
            dt=(dt:=0.05),
            random_seed=seed + global_seed_offset,
            roll_range_deg=0,
            pitch_range_deg=pr,
            vel_range=0,
            vel_slew_per_0p2s=1,
            roll_pert_stddev=0,
            pitch_pert_stddev=0.0,
            vel_pert_stddev=0
        )
        time = np.arange(0, duration, dt)
        
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)


def yaw(yaw_range_deg=90, duration_s=30.0, wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    """
    Run yaw experiments with pseudo-random binary sequence (PRBS) yaw setpoints.
    
    Args:
        yaw_range_deg: Maximum yaw amplitude in degrees (default: 90)
        duration_s: Duration of each experiment in seconds (default: 30)
        wheelbot_name: Name of the wheelbot to use
        surface: Surface type for metadata
        video_device: Video device for recording
    """
    for seed in range(10):
        print(f"Yaw experiment: range={yaw_range_deg}deg, seed={seed}")
        if not continue_skip_abort():
            continue
        
        path = next_log_number(f"data/yaw")
        dt = 0.05
        
        # Generate PRBS yaw setpoints
        yaw_setpoints = generate_yaw_prbs(
            duration_s=duration_s,
            dt=dt,
            yaw_range_deg=yaw_range_deg,
            min_duration_s=1.0,
            max_duration_s=2.0,
            random_seed=seed + global_seed_offset
        )
        
        # Display the raw PRBS signal first
        time_arr = np.arange(0, duration_s, dt)
        plt.figure(figsize=(14, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_arr, yaw_setpoints)
        plt.title(f"Yaw PRBS Setpoints (seed={seed})")
        plt.ylabel("Yaw [deg]")
        plt.grid(True)
        
        # Convert to delta angles for transmission
        yaw_deltas = convert_yaw_setpoints_to_deltas(yaw_setpoints)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_arr, yaw_deltas)
        plt.title("Yaw Delta Commands (sent to robot)")
        plt.ylabel("Yaw Delta [deg]")
        plt.xlabel("Time [s]")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Zero velocity, roll, pitch for yaw experiments
        velocity = np.zeros_like(yaw_deltas)
        roll = np.zeros_like(yaw_deltas)
        pitch = np.zeros_like(yaw_deltas)
        
        plot_and_run_with_repeat(
            velocity, roll, pitch, time_arr, dt,
            wheelbot_name=wheelbot_name,
            surface=surface,
            video_device=video_device,
            path=path,
            yaw_delta=yaw_deltas
        )


def velrollpitch(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    for seed in range(35,40):
        if not continue_skip_abort():
            continue
        dt = 0.05
        path = next_log_number(f"data/velocity_roll_pitch")
        duration=8*4
        velocity, roll, pitch = generate_setpoints(duration_s=duration, stabilize_every_n_seconds=4, stabilize_for=6, dt=dt, roll_range_deg=5, pitch_range_deg=3, vel_range=0.5, random_seed=seed + global_seed_offset)
        time = np.arange(0, duration, dt)
        
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)

def velrollpitch2(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    roll_ranges = np.linspace(1,10,3)
    for seed, rr in enumerate(roll_ranges):
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/velocity_roll")
        velocity, roll, pitch = generate_setpoints(
            duration_s=(duration:=30.0),
            stabilize_every_n_seconds=5,
            stabilize_for=5,
            dt=(dt:=0.05),
            random_seed=seed + global_seed_offset,
            roll_range_deg=rr,
            pitch_range_deg=0,
            vel_range=0,
            vel_slew_per_0p2s=1,
            roll_pert_stddev=0,
            pitch_pert_stddev=0,
            vel_pert_stddev=0
        )
        time = np.arange(0, duration, dt)
        velocity = build_linear_velocity_profile_for_other(roll, dt, target_velocity=0.5)
        
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)        

def lin(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    target_velocities = np.linspace(0.5,1.5,3)
    for seed, vel in enumerate(target_velocities):
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/velocity")
        velocity, roll, pitch, time, dt = linear_sequence(vel, accel_time=vel*2)
        
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)

def linwithlean(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    target_velocities = np.linspace(0.5, 1.5, 3)
    for seed, vel in enumerate(target_velocities):
        if not continue_skip_abort():
            continue
        path = next_log_number(f"data/velocity_pitch")
        # Scale pitch angle with velocity for more natural behavior
        pitch_accel = 3.0 + vel * 5.0  # Lean forward more at higher velocities
        pitch_decel = -(3.0 + vel * 5.0)  # Lean back more when decelerating from higher velocities
        velocity, roll, pitch, time, dt = linear_sequence_with_lean(
            target_velocity=vel, 
            const_time=1, 
            accel_time=vel,
            pitch_accel_deg=pitch_accel,
            pitch_decel_deg=pitch_decel
        )
        
        plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)

def lin2(wheelbot_name=default_wheelbot_name, surface=default_surface, video_device=default_video_device):
    target_velocities = np.linspace(0.5,1,3)
    dirs = [-1, +1]
    for seed, vel in enumerate(target_velocities):
        for dir in dirs:
            if not continue_skip_abort():
                continue
            path = next_log_number(f"data/velocity_roll")
            velocity, time, dt = linear_velocity_sequence(target_velocity=1.0, const_time=1, accel_time=1.0)
            roll = np.zeros_like(velocity)       
            accel_samples = int(1.0 / dt)
            const_samples = int(1. / dt)
            decel_samples = int(1.0 / dt)        
            pulse_duration = 0.2
            pulse_samples = int(pulse_duration / dt)
            pause_duration = 0.4
            pause_samples = int(pause_duration / dt)
            initial_delay = 0.2  # Time at constant velocity before first pulse
            initial_delay_samples = int(initial_delay / dt)
            pulse_amplitudes = [dir*10, -dir*10]  # Alternating positive and negative
            current_idx = accel_samples + initial_delay_samples
            for amplitude in pulse_amplitudes:
                if current_idx + pulse_samples > accel_samples + const_samples:
                    break
                roll[current_idx:current_idx + pulse_samples] = amplitude
                current_idx += pulse_samples + pause_samples
            
            pitch = np.zeros_like(velocity)
            plot_and_run_with_repeat(velocity, roll, pitch, time, dt, wheelbot_name=wheelbot_name, surface=surface, video_device=video_device, path=path)


def consolidate(input_dataset_paths: list[str] | str, output_dataset_path: str):
    """
    Consolidate multiple dataset directories into a single unified dataset.
    This function takes one or more input dataset paths and combines them into a single
    output dataset, renumbering experiments sequentially within each group. It handles
    datasets organized by groups (e.g., velocity, roll, pitch) and ensures all required
    files for each experiment are present before copying.
    Args:
        input_dataset_paths: Either a single directory path containing multiple dataset
            subdirectories, or a list of dataset directory paths to consolidate.
        output_dataset_path: Path where the consolidated dataset will be created.
    Raises:
        None: Prints warnings and prompts user for confirmation if output path exists.
    Returns:
        None
    Notes:
        - Each experiment must have all required files (.csv, .log, .meta, .mp4, .pkl,
          .preview.pdf, .setpoints.pdf) to be included in the consolidated dataset.
        - Experiments are renumbered sequentially within each group, starting from 0.
        - If output_dataset_path exists, the user is prompted to delete it before proceeding.
    """
    if isinstance(input_dataset_paths, str):
        input_dataset_paths = [
            os.path.join(input_dataset_paths, d) 
            for d in os.listdir(input_dataset_paths)
            if os.path.isdir(os.path.join(input_dataset_paths, d))
        ]
        
        if os.path.exists(output_dataset_path):
            print(f"Warning: Output dataset path already exists: {output_dataset_path}")
            response = input(f"Do you want to delete '{output_dataset_path}' and recreate it? (y/N): ").strip().lower()
            if response == 'y':
                shutil.rmtree(output_dataset_path)
                print(f"Deleted: {output_dataset_path}")
            else:
                print("Aborting.")
                return
    os.makedirs(output_dataset_path, exist_ok=True)
    
    next_number = {}
    for input_path in input_dataset_paths:
        if not os.path.isdir(input_path):
            print(f"Skipping non-directory: {input_path}")
            continue
        
        print(f"Processing: {input_path}")
        
        # Iterate through groups (velocity, roll, pitch, etc.)
        for group_name in os.listdir(input_path):
            group_path = os.path.join(input_path, group_name)
            if not os.path.isdir(group_path):
                continue
            
            # Initialize next number for this group if not seen before
            if group_name not in next_number:
                next_number[group_name] = 0
            
            # Create output group directory
            output_group_path = os.path.join(output_dataset_path, group_name)
            os.makedirs(output_group_path, exist_ok=True)
            
            # Find all experiment numbers in this group
            experiment_numbers = set()
            for filename in os.listdir(group_path):
                if '.' in filename:
                    number_str = filename.split('.')[0]
                    if number_str.isdigit():
                        experiment_numbers.add(int(number_str))
            
            # Process each experiment
            for exp_num in sorted(experiment_numbers):
                # Check if all required files exist
                required_extensions = ['.csv', '.log', '.meta', '.mp4', '.pkl', '.preview.pdf', '.setpoints.pdf']
                all_files_present = all(
                    os.path.exists(os.path.join(group_path, f"{exp_num}{ext}"))
                    for ext in required_extensions
                )
                
                if not all_files_present:
                    print(f"  Skipping incomplete experiment {group_name}/{exp_num}")
                    continue
                
                # Copy all files with new numbering
                new_num = next_number[group_name]
                for ext in required_extensions:
                    src = os.path.join(group_path, f"{exp_num}{ext}")
                    dst = os.path.join(output_group_path, f"{new_num}{ext}")
                    shutil.copy2(src, dst)
                
                print(f"  Copied {group_name}/{exp_num} -> {group_name}/{new_num}")
                next_number[group_name] += 1

    print(f"\nConsolidation complete. Output: {output_dataset_path}")        

if __name__ == "__main__":
    fire.Fire({
        "vel": vel,
        "roll": roll,
        "roll_max": roll_max,
        "pitch": pitch,
        "yaw": yaw,
        "velrollpitch": velrollpitch,
        "velrollpitch2": velrollpitch2,
        "lin": lin,
        "linwithlean": linwithlean,
        "lin2": lin2,
        "consolidate": consolidate,
        "consolidate_archive_to_data": lambda: consolidate("archive", "data_consolidated"),
    })
