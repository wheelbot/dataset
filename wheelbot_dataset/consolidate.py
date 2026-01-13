#!/usr/bin/env python3
"""
Consolidation and analysis tools for wheelbot datasets.

This module provides functions for consolidating multiple datasets into one
and computing statistics over datasets.
"""

import os
import shutil
import json
import pandas as pd
import numpy as np
from typing import Union
import fire
from wheelbot_dataset.usage.dataset import Dataset


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


def statistics(dataset_path: str, cutoff_seconds: float = 7.0):
    """
    Compute statistics over a wheelbot dataset and output a LaTeX table.
    
    This function analyzes a dataset and computes per-group statistics including:
    - Number of trajectories (experiments) in each group
    - Total duration of all trajectories (subtracting cutoff_seconds from each trajectory)
    - Number of crashes (trajectories marked as "experiment_status": "failed")
    
    Groups are mapped to their controller types and reference types, and roll/roll_max
    are combined into a single "Roll" category.
    
    Args:
        dataset_path: Path to the root directory of the dataset.
        cutoff_seconds: Number of seconds to subtract from each trajectory's duration
                       (default: 7.0). This accounts for beginning and end segments
                       that would typically be cut off by users.
    
    Returns:
        None: Prints statistics and LaTeX table to stdout.
    """
    print(f"\nDataset Statistics for: {dataset_path}")
    print("=" * 80)
    
    # Load the dataset
    try:
        ds = Dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Define mapping from group names to display names, controllers, and reference types
    group_mapping = {
        'pitch': {'name': 'Pitch', 'controller': 'LQR', 'reference': 'PRBS'},
        'roll': {'name': 'Roll', 'controller': 'LQR', 'reference': 'PRBS'},
        'roll_max': {'name': 'Roll', 'controller': 'LQR', 'reference': 'PRBS'},
        'velocity_roll': {'name': 'Vel + Roll', 'controller': 'LQR', 'reference': 'PRBS'},
        'velocity_pitch': {'name': 'Vel + Pitch', 'controller': 'LQR', 'reference': 'PRBS'},
        'velocity': {'name': 'Velocity', 'controller': 'LQR', 'reference': 'PRBS'},
        'yaw': {'name': 'Yaw Random', 'controller': 'AMPC', 'reference': 'PRBS'},
        'yaw_circle': {'name': 'Yaw Circles', 'controller': 'AMPC', 'reference': 'Geometric'},
        'yaw_figure_eight': {'name': 'Yaw Eight', 'controller': 'AMPC', 'reference': 'Geometric'},
    }
    
    # Collect statistics by display name (combining roll and roll_max)
    stats = {}
    
    for group_name in sorted(ds.groups.keys()):
        if group_name not in group_mapping:
            continue
            
        group = ds.groups[group_name]
        display_info = group_mapping[group_name]
        display_name = display_info['name']
        
        # Initialize stats for this display name if not exists
        if display_name not in stats:
            stats[display_name] = {
                'controller': display_info['controller'],
                'reference': display_info['reference'],
                'num_trajs': 0,
                'duration': 0.0,
                'crashes': 0
            }
        
        # Accumulate statistics
        for exp in group.experiments:
            stats[display_name]['num_trajs'] += 1
            
            # Get duration from the CSV data
            if len(exp.data) > 0:
                duration = exp.data.index[-1] - exp.data.index[0]
                adjusted_duration = max(0, duration - cutoff_seconds)
                stats[display_name]['duration'] += adjusted_duration
            
            # Check if experiment failed
            if exp.meta.get("experiment_status") == "failed":
                stats[display_name]['crashes'] += 1
    
    # Print individual group statistics
    for display_name in ['Pitch', 'Roll', 'Velocity', 'Vel + Roll', 'Vel + Pitch', 'Yaw Random', 'Yaw Circles', 'Yaw Eight']:
        if display_name in stats:
            info = stats[display_name]
            print(f"\n{display_name}")
            print("-" * 40)
            print(f"Number of trajectories: {info['num_trajs']}")
            print(f"Total duration (after {cutoff_seconds}s cutoff per trajectory): {info['duration']:.0f} seconds")
            print(f"Crashes: {info['crashes']}")
    
    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table")
    print("=" * 80)
    print()
    
    # Define the order of rows in the table
    row_order = ['Pitch', 'Roll', 'Vel + Roll', 'Vel + Pitch', 'Yaw Random', 'Yaw Circles', 'Yaw Eight']
    
    # Calculate totals for the "All" row
    total_trajectories = sum(info['num_trajs'] for info in stats.values())
    total_duration_all = sum(info['duration'] for info in stats.values())
    total_crashes = sum(info['crashes'] for info in stats.values())
    
    for display_name in row_order:
        if display_name in stats:
            info = stats[display_name]
            duration_min = info['duration'] / 60.0
            print(f"      {display_name:18} & {info['controller']:10} & {info['reference']:14} & "
                  f"{info['num_trajs']:<15} & {duration_min:<23.1f} & {info['crashes']:<11} \\\\")
        else:
            # Use placeholder values if group doesn't exist
            controller = group_mapping.get(display_name.lower().replace(' ', '_'), {}).get('controller', 'LQR')
            reference = group_mapping.get(display_name.lower().replace(' ', '_'), {}).get('reference', 'PRBS')
            print(f"      {display_name:18} & {controller:10} & {reference:14} & "
                  f"{'--':<15} & {'--':<23} & {'--':<11} \\\\")
    
    # Add the "All" row with totals
    print(f"      \\midrule")
    total_duration_min = total_duration_all / 60.0
    print(f"      {'All':18} & {'':10} & {'':14} & "
          f"{total_trajectories:<15} & {total_duration_min:<23.1f} & {total_crashes:<11} \\\\")
    
    print()
    
    # Overall statistics
    print("=" * 80)
    print("Overall Statistics")
    print("-" * 40)
    
    print(f"Total trajectories: {total_trajectories}")
    print(f"Total duration (after {cutoff_seconds}s cutoff per trajectory): {total_duration_all:.0f} seconds ({total_duration_all/60:.1f} minutes)")
    print(f"Total crashes: {total_crashes}/{total_trajectories} ({total_crashes/total_trajectories*100:.1f}%)")
    print()


def updaterates(dataset_path: str, group_name: str, index: int):
    """
    Analyze true update rates of data columns in a specific experiment.
    
    Calculates the effective update rate for each column by detecting how many
    consecutive rows have the same value, then computing the actual rate at which
    values change.
    
    The update rate is calculated by:
    1. Detecting runs of consecutive identical values for each column
    2. Computing the average run length (how many rows have the same value)
    3. Calculating the update rate as: sampling_rate / avg_run_length
    
    Args:
        dataset_path: Path to the root directory of the dataset.
        group_name: Name of the experiment group (e.g., "pitch", "roll", "yaw").
        index: Index of the experiment within the group to analyze.
    
    Returns:
        None: Prints update rate analysis to stdout.
        
    Example:
        >>> updaterates("data", "pitch", 4)
        Analyzing experiment: data/pitch/4.csv
        ...
    """
    # Load the dataset
    try:
        ds = Dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load the specified group
    try:
        group = ds.load_group(group_name)
    except KeyError:
        print(f"Error: Group '{group_name}' not found in dataset.")
        print(f"Available groups: {', '.join(sorted(ds.groups.keys()))}")
        return
    
    # Get the specified experiment
    if index < 0 or index >= len(group.experiments):
        print(f"Error: Index {index} out of range. Group '{group_name}' has {len(group.experiments)} experiments (0-{len(group.experiments)-1}).")
        return
    
    experiment = group[index]
    df = experiment.data
    
    # Calculate the nominal sampling rate from the time index
    time_diffs = np.diff(df.index)
    avg_dt = np.mean(time_diffs)
    nominal_rate = 1.0 / avg_dt if avg_dt > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Analyzing experiment: {experiment.csv_path}")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Time range: {df.index[0]:.3f} to {df.index[-1]:.3f} seconds")
    print(f"Duration: {df.index[-1] - df.index[0]:.3f} seconds")
    print(f"Average time step: {avg_dt*1000:.3f} ms")
    print(f"Nominal sampling rate: {nominal_rate:.2f} Hz")
    print(f"\n{'='*80}")
    print(f"Column Update Rate Analysis")
    print(f"{'='*80}")
    print(f"{'Column':<50} {'Avg Run':<12} {'Update Rate':<15} {'vs Nominal'}")
    print(f"{'-'*50} {'-'*12} {'-'*15} {'-'*12}")
    
    update_rates = {}
    
    for col in df.columns:
        # Get the column data as numpy array
        data = df[col].values
        
        # Find where values change
        # Compare each value with the next one
        changes = np.concatenate([[True], data[1:] != data[:-1]])
        
        # Find indices where changes occur
        change_indices = np.where(changes)[0]
        
        # Calculate run lengths (number of consecutive identical values)
        if len(change_indices) > 1:
            run_lengths = np.diff(change_indices)
            avg_run_length = np.mean(run_lengths)
        elif len(change_indices) == 1:
            # Only one change or all values are the same
            avg_run_length = len(data)
        else:
            # No changes at all
            avg_run_length = len(data)
        
        # Calculate the update rate
        # Update rate = nominal_rate / avg_run_length
        update_rate = nominal_rate / avg_run_length if avg_run_length > 0 else 0
        
        update_rates[col] = update_rate
        
        # Calculate ratio to nominal rate
        ratio = update_rate / nominal_rate if nominal_rate > 0 else 0
        
        # Determine if this is likely a low-rate signal
        if avg_run_length > 1.5:
            marker = " *"
        else:
            marker = ""
        
        print(f"{col:<50} {avg_run_length:>10.2f}   {update_rate:>10.2f} Hz   {ratio:>8.2%}{marker}")
    
    # Analyze by update rate
    threshold_ratio = 0.5
    high_rate = {}
    low_rate = {}
    
    for col, rate in update_rates.items():
        ratio = rate / nominal_rate if nominal_rate > 0 else 0
        if ratio >= threshold_ratio:
            high_rate[col] = rate
        else:
            low_rate[col] = rate
    
    print(f"\n{'='*80}")
    print("* Indicates signals with update rate significantly below nominal rate")
    print(f"{'='*80}\n")
    
    print(f"Summary Statistics:")
    print(f"{'-'*80}")
    print(f"High-rate signals (>= {threshold_ratio*100:.0f}% of nominal): {len(high_rate)}")
    print(f"Low-rate signals (< {threshold_ratio*100:.0f}% of nominal): {len(low_rate)}")
    
    if low_rate:
        print(f"\nLow-rate signals:")
        for col, rate in sorted(low_rate.items(), key=lambda x: x[1]):
            print(f"  {col:<48} {rate:>8.2f} Hz")
    
    print()


if __name__ == "__main__":
    fire.Fire({
        "consolidate": consolidate,
        "statistics": statistics,
        "updaterates": updaterates,
    })
