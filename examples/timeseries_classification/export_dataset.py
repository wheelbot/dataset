import numpy as np
import pickle
import os

from wheelbot_dataset import (
    Dataset,
    to_prediction_dataset,
)


def export_surface_classification_dataset():
    """
    Export IMU timeseries data for surface classification.
    
    This creates a dataset for training a transformer model to classify surfaces
    (black_pvc, concrete, felt) based on gyroscope and accelerometer data.
    """
    # Load the dataset - only yaw-related groups
    ds = Dataset("../../data")
    
    # Filter to only include yaw-related groups
    yaw_groups = ["yaw", "yaw_circle", "yaw_figure_eight"]
    filtered_groups = {k: v for k, v in ds.groups.items() if k in yaw_groups}
    ds.groups = filtered_groups
    
    print(f"Using groups: {list(ds.groups.keys())}")
    print()
    
    # Define IMU fields to export (gyro and accel for sensors 0-3)
    fields_states = [
        "/gyro0/x", "/gyro0/y", "/gyro0/z",
        "/accel0/x", "/accel0/y", "/accel0/z",
    ]

    
    # No actions needed for classification
    fields_actions = []
    
    # Define preprocessing (same cut_by_condition and cut_time as in VRP example)
    dt = 0.01
    def cut_and_filter_fn(exp):
        return (
            exp
            .cut_by_condition(
                start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
                end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0,
            )
            .resample(dt=dt)
            .cut_time(start=2.0, end=2.0)
        )
    
    # Surface labels to export
    surfaces = ["black_pvc", "concrete", "gray_felt"]
    
    # Collect data for each surface (unbalanced)
    all_timeseries = []
    all_labels = []
    surface_samples = {}  # Store samples per surface for balancing
    
    for surface_idx, surface in enumerate(surfaces):
        print(f"Processing surface: {surface}...")
        
        # Filter experiments by surface and successful status
        filtered_ds = ds.map(
            lambda exp: exp.filter_by_metadata(
                experiment_status="success",
                surface=surface
            )
        ).map(cut_and_filter_fn)
        
        # Convert to prediction dataset format
        # N_future=500, skip_N=500 for non-overlapping windows
        states, actions, nextstates, _ = to_prediction_dataset(
            filtered_ds,
            fields_states=fields_states,
            fields_actions=fields_actions,
            N_past=int(0.5/dt),
            skip_N=int(0.5/dt)
        )
        
        print(f"  Surface '{surface}': {states.shape[0]} samples (before balancing)")
        
        # Store for balancing
        surface_samples[surface_idx] = states
    
    # Balance the dataset: find the median sample count
    sample_counts = [s.shape[0] for s in surface_samples.values()]
    target_samples = int(np.median(sample_counts))
    print(f"\nBalancing dataset to approximately {target_samples} samples per class...")
    
    # Subsample or keep all samples to balance classes
    np.random.seed(42)  # For reproducible sampling
    for surface_idx, surface in enumerate(surfaces):
        states = surface_samples[surface_idx]
        n_samples = states.shape[0]
        
        if n_samples > target_samples:
            # Randomly subsample to target_samples
            indices = np.random.choice(n_samples, target_samples, replace=False)
            states = states[indices]
            print(f"  {surface}: Subsampled from {n_samples} to {states.shape[0]} samples")
        else:
            print(f"  {surface}: Keeping all {n_samples} samples")
        
        all_timeseries.append(states)
        all_labels.extend([surface_idx] * states.shape[0])
    
    # Concatenate all surfaces
    timeseries = np.concatenate(all_timeseries, axis=0)
    labels = np.array(all_labels)
    
    print(f"\nTotal dataset shape: {timeseries.shape}")
    print(f"Total labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Shuffle the data
    print("\nShuffling data...")
    indices = np.arange(len(timeseries))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    timeseries = timeseries[indices]
    labels = labels[indices]
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(timeseries))
    
    train_timeseries = timeseries[:split_idx]
    train_labels = labels[:split_idx]
    
    test_timeseries = timeseries[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"\nTrain set: {train_timeseries.shape[0]} samples")
    print(f"  Label distribution: {np.bincount(train_labels)}")
    print(f"Test set: {test_timeseries.shape[0]} samples")
    print(f"  Label distribution: {np.bincount(test_labels)}")
    
    # Save to pickle
    os.makedirs("dataset", exist_ok=True)
    output_path = "dataset/surface_classification_dataset.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "train_timeseries": train_timeseries,
            "train_labels": train_labels,
            "test_timeseries": test_timeseries,
            "test_labels": test_labels,
            "fields_states": fields_states,
            "surfaces": surfaces,
            "n_classes": len(surfaces),
        }, f)
    
    print(f"\nDataset saved to: {output_path}")
    

def export_robot_classification_dataset():
    """
    Export IMU timeseries data for robot classification.
    
    This creates a dataset for training a transformer model to classify robots
    (wheelbot-beta-1, wheelbot-beta-2, wheelbot-beta-3) based on gyroscope 
    and accelerometer data. Only uses black_pvc surface.
    """
    # Load the dataset - only yaw-related groups
    ds = Dataset("../../data")
    
    # Filter to only include yaw-related groups
    yaw_groups = ["yaw", "yaw_circle", "yaw_figure_eight"]
    filtered_groups = {k: v for k, v in ds.groups.items() if k in yaw_groups}
    ds.groups = filtered_groups
    
    print(f"Using groups: {list(ds.groups.keys())}")
    print()
    
    # Define IMU fields to export (gyro and accel for sensors 0-3)
    fields_states = [
        "/gyro0/x", "/gyro0/y", "/gyro0/z",
        "/accel0/x", "/accel0/y", "/accel0/z",
    ]
    
    # No actions needed for classification
    fields_actions = []
    
    # Define preprocessing
    dt = 0.01
    def cut_and_filter_fn(exp):
        return (
            exp
            .cut_by_condition(
                start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
                end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0,
            )
            .resample(dt=dt)
            .cut_time(start=2.0, end=2.0)
        )
    
    # Robot labels to export
    robots = ["wheelbot-beta-1", "wheelbot-beta-2", "wheelbot-beta-3"]
    
    # Collect data for each robot (unbalanced)
    all_timeseries = []
    all_labels = []
    robot_samples = {}
    
    for robot_idx, robot in enumerate(robots):
        print(f"Processing robot: {robot}...")
        
        # Filter experiments by robot, black_pvc surface, and successful status
        filtered_ds = ds.map(
            lambda exp: exp.filter_by_metadata(
                experiment_status="success",
                surface="black_pvc",
                wheelbot=robot
            )
        ).map(cut_and_filter_fn)
        
        # Convert to prediction dataset format
        states, actions, nextstates, _ = to_prediction_dataset(
            filtered_ds,
            fields_states=fields_states,
            fields_actions=fields_actions,
            N_past=int(0.5/dt),
            skip_N=int(0.5/dt)
        )
        
        print(f"  Robot '{robot}': {states.shape[0]} samples (before balancing)")
        robot_samples[robot_idx] = states
    
    # Balance the dataset
    sample_counts = [s.shape[0] for s in robot_samples.values()]
    target_samples = int(np.median(sample_counts))
    print(f"\nBalancing dataset to approximately {target_samples} samples per class...")
    
    np.random.seed(42)
    for robot_idx, robot in enumerate(robots):
        states = robot_samples[robot_idx]
        n_samples = states.shape[0]
        
        if n_samples > target_samples:
            indices = np.random.choice(n_samples, target_samples, replace=False)
            states = states[indices]
            print(f"  {robot}: Subsampled from {n_samples} to {states.shape[0]} samples")
        else:
            print(f"  {robot}: Keeping all {n_samples} samples")
        
        all_timeseries.append(states)
        all_labels.extend([robot_idx] * states.shape[0])
    
    # Concatenate all robots
    timeseries = np.concatenate(all_timeseries, axis=0)
    labels = np.array(all_labels)
    
    print(f"\nTotal dataset shape: {timeseries.shape}")
    print(f"Total labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Shuffle the data
    print("\nShuffling data...")
    indices = np.arange(len(timeseries))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    timeseries = timeseries[indices]
    labels = labels[indices]
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(timeseries))
    
    train_timeseries = timeseries[:split_idx]
    train_labels = labels[:split_idx]
    
    test_timeseries = timeseries[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"\nTrain set: {train_timeseries.shape[0]} samples")
    print(f"  Label distribution: {np.bincount(train_labels)}")
    print(f"Test set: {test_timeseries.shape[0]} samples")
    print(f"  Label distribution: {np.bincount(test_labels)}")
    
    # Save to pickle
    os.makedirs("dataset", exist_ok=True)
    output_path = "dataset/robot_classification_dataset.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "train_timeseries": train_timeseries,
            "train_labels": train_labels,
            "test_timeseries": test_timeseries,
            "test_labels": test_labels,
            "fields_states": fields_states,
            "robots": robots,
            "n_classes": len(robots),
        }, f)
    
    print(f"\nDataset saved to: {output_path}")


def export_group_classification_dataset():
    """
    Export IMU timeseries data for group classification.
    
    This creates a dataset for training a transformer model to classify experiment groups
    (yaw/yaw_circle/yaw_figure_eight vs yaw_human) based on gyroscope and accelerometer data.
    Only uses black_pvc surface.
    """
    # Load the dataset
    ds = Dataset("../../data")
    
    # Define IMU fields to export (gyro and accel for sensors 0-3)
    fields_states = [
        "/gyro0/x", "/gyro0/y", "/gyro0/z",
        "/accel0/x", "/accel0/y", "/accel0/z",
    ]
    
    # No actions needed for classification
    fields_actions = []
    
    # Define preprocessing
    dt = 0.01
    def cut_and_filter_fn(exp):
        result = exp.cut_by_condition(
            start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
            end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0,
        )
        # Check if result is None or has empty data
        if result is None or len(result.data) == 0:
            return None
        return result.resample(dt=dt).cut_time(start=2.0, end=2.0)
    
    # Group categories
    group_categories = {
        "autonomous": ["yaw", "yaw_circle", "yaw_figure_eight"],
        "human": ["yaw_human"]
    }
    
    all_timeseries = []
    all_labels = []
    category_samples = {}
    
    for category_idx, (category_name, group_list) in enumerate(group_categories.items()):
        print(f"Processing category: {category_name} (groups: {group_list})...")
        
        # Filter to only include specified groups
        filtered_groups = {k: v for k, v in ds.groups.items() if k in group_list}
        ds_filtered = Dataset.__new__(Dataset)
        ds_filtered.root = ds.root
        ds_filtered.groups = filtered_groups
        
        # Filter experiments by black_pvc surface and successful status
        filtered_ds = ds_filtered.map(
            lambda exp: exp.filter_by_metadata(
                experiment_status="success",
                surface="black_pvc",
                wheelbot="wheelbot-beta-1"
            )
        ).map(cut_and_filter_fn)
        
        # Convert to prediction dataset format
        states, actions, nextstates, _ = to_prediction_dataset(
            filtered_ds,
            fields_states=fields_states,
            fields_actions=fields_actions,
            N_past=int(0.5/dt),
            skip_N=int(0.5/dt)
        )
        
        print(f"  Category '{category_name}': {states.shape[0]} samples (before balancing)")
        category_samples[category_idx] = states
    
    # Balance the dataset
    sample_counts = [s.shape[0] for s in category_samples.values()]
    target_samples = int(np.median(sample_counts))
    print(f"\nBalancing dataset to approximately {target_samples} samples per class...")
    
    np.random.seed(42)
    category_names = list(group_categories.keys())
    for category_idx, category_name in enumerate(category_names):
        states = category_samples[category_idx]
        n_samples = states.shape[0]
        
        if n_samples > target_samples:
            indices = np.random.choice(n_samples, target_samples, replace=False)
            states = states[indices]
            print(f"  {category_name}: Subsampled from {n_samples} to {states.shape[0]} samples")
        else:
            print(f"  {category_name}: Keeping all {n_samples} samples")
        
        all_timeseries.append(states)
        all_labels.extend([category_idx] * states.shape[0])
    
    # Concatenate all categories
    timeseries = np.concatenate(all_timeseries, axis=0)
    labels = np.array(all_labels)
    
    print(f"\nTotal dataset shape: {timeseries.shape}")
    print(f"Total labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Shuffle the data
    print("\nShuffling data...")
    indices = np.arange(len(timeseries))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    timeseries = timeseries[indices]
    labels = labels[indices]
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(timeseries))
    
    train_timeseries = timeseries[:split_idx]
    train_labels = labels[:split_idx]
    
    test_timeseries = timeseries[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"\nTrain set: {train_timeseries.shape[0]} samples")
    print(f"  Label distribution: {np.bincount(train_labels)}")
    print(f"Test set: {test_timeseries.shape[0]} samples")
    print(f"  Label distribution: {np.bincount(test_labels)}")
    
    # Save to pickle
    os.makedirs("dataset", exist_ok=True)
    output_path = "dataset/group_classification_dataset.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "train_timeseries": train_timeseries,
            "train_labels": train_labels,
            "test_timeseries": test_timeseries,
            "test_labels": test_labels,
            "fields_states": fields_states,
            "categories": category_names,
            "n_classes": len(category_names),
        }, f)
    
    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("EXPORTING SURFACE CLASSIFICATION DATASET")
    print("=" * 80)
    export_surface_classification_dataset()
    
    print("\n" + "=" * 80)
    print("EXPORTING ROBOT CLASSIFICATION DATASET")
    print("=" * 80)
    export_robot_classification_dataset()
    
    print("\n" + "=" * 80)
    print("EXPORTING GROUP CLASSIFICATION DATASET")
    print("=" * 80)
    export_group_classification_dataset()
