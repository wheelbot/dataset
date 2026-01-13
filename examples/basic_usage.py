#!/usr/bin/env python3
"""
Example usage script for the wheelbot-dataset package.

This script demonstrates how to load, filter, and process wheelbot experiment data
using the wheelbot_dataset package.

Usage:
    python -m wheelbot_dataset.example_usage
"""

import pickle
from wheelbot_dataset import (
    Dataset,
    plot_timeseries,
    plot_histograms,
    to_prediction_dataset,
)


def example_usage():
    # Load a dataset
    ds = Dataset("data")

    # Dataset contains folders as groups
    group = ds.load_group("velocity_roll_pitch")
    
    # Groups contain lists of experiments
    exp = group[0]
    
    # Remove before standup and standup, lay down and after laydown from experiment
    exp_cut = (
        exp.cut_by_condition(
            start_condition = lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
            end_condition   = lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0
        )
        .cut_time(start=2.0, end=2.0)
    )

    # Implement some acausal filter
    import scipy.signal as signal
    lowpass_cutoff_hz=50
    def my_lowpass(df):
        b, a = signal.butter(4, 2*(lowpass_cutoff_hz)/1000, btype="low", analog=False)
        df2 = df.copy()
        for col in df.columns:
            df2[col] = signal.filtfilt(b, a, df[col])
        return df2
    
    # Filter and downsample a single experiment (original data at 1kHz)
    exp_filtered = (
        exp_cut.apply_filter(my_lowpass)
        .resample(dt=(dt:=0.01))
    )

    # Print fields and metadata of an experiment
    print(exp_filtered.data.head())
    print(exp_filtered.meta)
    
    # Plot an experiment
    plot_fields = {
        "Gyro 0": ["/gyro0/x", "/gyro0/y", "/gyro0/z"],
        "Gyro 1": ["/gyro1/x", "/gyro1/y", "/gyro1/z"],
        "Gyro 2": ["/gyro2/x", "/gyro2/y", "/gyro2/z"],
        "Gyro 3": ["/gyro3/x", "/gyro3/y", "/gyro3/z"],
        "Accel 0": ["/accel0/x", "/accel0/y", "/accel0/z"],
        "Accel 1": ["/accel1/x", "/accel1/y", "/accel1/z"],
        "Accel 2": ["/accel2/x", "/accel2/y", "/accel2/z"],
        "Accel 3": ["/accel3/x", "/accel3/y", "/accel3/z"],
        "YPR": ["/q_yrp/yaw", "/q_yrp/roll", "/q_yrp/pitch"],
        "YPR Vel": ["/dq_yrp/yaw_vel", "/dq_yrp/roll_vel", "/dq_yrp/pitch_vel"],
        "Wheel Position": ["/q_DR/drive_wheel", "/q_DR/reaction_wheel"],
        "Wheel Velocity": ["/dq_DR/drive_wheel", "/dq_DR/reaction_wheel"],
        "Wheel Acceleration": ["/ddq_DR/drive_wheel", "/ddq_DR/reaction_wheel"],
        "Commands": ["/tau_DR_command/drive_wheel", "/tau_DR_command/reaction_wheel"],
        "Setpoint Euler": ["/setpoint/yaw", "/setpoint/roll", "/setpoint/pitch"],
        "Setpoint Rates": ["/setpoint/yaw_rate", "/setpoint/roll_rate", "/setpoint/pitch_rate"],
        "Setpoint Wheels": [
            "/setpoint/driving_wheel_angle",
            "/setpoint/driving_wheel_angular_velocity",
            "/setpoint/balancing_wheel_angle",
            "/setpoint/balancing_wheel_angular_velocity",
        ],
        "Vicon Position": ["/vicon_position/x", "/vicon_position/y", "/vicon_position/z"],
        "Battery": ["battery_voltage"],
    }
    plot_timeseries(
        experiments=exp_cut,
        groups=plot_fields,
        pdf_path="test/exp_unfiltered.pdf"
    )
    plot_timeseries(
        experiments=exp_filtered,
        groups=plot_fields,
        pdf_path="test/exp_filtered.pdf"
    )
    
    # Define the same filters to all experiments in a group or dataset
    cut_and_filter_fn = lambda exp: (
        exp
        .cut_by_condition(
            start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
            end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0,
        )
        .cut_time(start=2.0, end=2.0)
        .apply_filter(my_lowpass)
        .resample(dt=0.01)
    )
    
    # Apply filter function to a group
    # filter_by_metadata needs to be applied separately or as the last filter (inefficient)
    group_filtered = (
        group
        .map(lambda exp: exp.filter_by_metadata(experiment_status="success"))
        .map(cut_and_filter_fn)
    )
    
    # Apply filter function to an entire dataset
    # you can also just map generic lambdas that remove experiments by returning None
    ds_filtered = (
        ds
        .map(lambda exp: exp.filter_by_metadata(experiment_status="success"))
        .map(cut_and_filter_fn)
        .map(lambda exp: exp if "battery/voltage" in exp.data.columns else None)
    )
    
    # Plot histograms of all fields
    plot_histograms(exp_filtered,   "test/exp0_histograms.pdf" )
    plot_histograms(group_filtered, "test/velocity_group_hist.pdf" )
    plot_histograms(ds_filtered,    "test/full_dataset_hist.pdf" )
    
    # Export an experiment to numpy (filter functions and metadata will not work anymore)
    columns = ["time"]+list(exp_filtered.data.columns)
    exp_numpy = exp_filtered.to_numpy(columns)
    group_numpy = group_filtered.map(lambda exp: exp.to_numpy(columns))
    ds_numpy = ds_filtered.map(lambda exp: exp.to_numpy(columns))
    
    # Convert and export states, actions, nextstates for neural network training
    fields_states = [
        "/q_yrp/roll","/q_yrp/pitch",
        "/dq_yrp/yaw_vel","/dq_yrp/roll_vel","/dq_yrp/pitch_vel",
        "/dq_DR/drive_wheel","/dq_DR/reaction_wheel",
        "/ddq_DR/drive_wheel","/ddq_DR/reaction_wheel",
        "battery/voltage"
    ]
    fields_actions = [
        "/tau_DR_command/drive_wheel","/tau_DR_command/reaction_wheel",
    ]
    states, actions, nextstates, _ = to_prediction_dataset(
        ds.map(cut_and_filter_fn),
        fields_states=fields_states,
        fields_actions=fields_actions        
    )
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next states shape: {nextstates.shape}")
    with open("dataset/dataset_1_step.pkl", "wb") as f:
        pickle.dump({
            "states": states,
            "actions": actions,
            "nextstates": nextstates,
            "states_labels": fields_states,
            "actions_labels": fields_actions,
            "dt": dt,
            "lowpass_cutoff_hz": lowpass_cutoff_hz,
            "filter_type": "scipy.signal.butter"
        }, f)
        
    states, actions, nextstates, _ = to_prediction_dataset(
        ds.map(cut_and_filter_fn),
        fields_states=fields_states,
        fields_actions=fields_actions,
        N_future=100
    )
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next states shape: {nextstates.shape}")
    with open("dataset/dataset_100_step.pkl", "wb") as f:
        pickle.dump({
            "states": states,
            "actions": actions,
            "nextstates": nextstates,
            "states_labels": fields_states,
            "actions_labels": fields_actions,
            "dt": dt,
            "lowpass_cutoff_hz": lowpass_cutoff_hz,
            "filter_type": "scipy.signal.butter"
        }, f)


if __name__ == "__main__":
    example_usage()
