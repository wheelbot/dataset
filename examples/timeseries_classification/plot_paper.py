"""
Create publication-quality figure for timeseries classification results.

Generates a 3.5-inch wide figure with three subplots showing accuracy vs sequence length
for surface, robot, and group classification tasks.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication-quality parameters
mpl.rcParams['font.size'] = 6
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 5
mpl.rcParams['figure.titlesize'] = 6
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.markersize'] = 3.5


def load_results(task):
    """Load results for a specific task."""
    try:
        with open(f"results/{task}_sequence_length_comparison.pkl", "rb") as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        print(f"Warning: Results file for task '{task}' not found.")
        return None


def load_dataset_info(task):
    """Load dataset to get label distribution for chance-level accuracy."""
    dataset_files = {
        "surface": "dataset/surface_classification_dataset.pkl",
        "robot": "dataset/robot_classification_dataset.pkl",
        "group": "dataset/group_classification_dataset.pkl"
    }
    
    try:
        with open(dataset_files[task], "rb") as f:
            data = pickle.load(f)
        
        # Calculate chance-level accuracy from test set label distribution
        test_labels = data["test_labels"]
        label_counts = np.bincount(test_labels)
        total = len(test_labels)
        
        # Chance level = sum of squared proportions (expected accuracy when guessing randomly)
        chance_level = np.sum((label_counts / total) ** 2)
        
        return chance_level, data["n_classes"]
    except FileNotFoundError:
        print(f"Warning: Dataset file for task '{task}' not found.")
        return None, None


def create_paper_figure():
    """Create publication-quality figure with three subplots."""
    
    tasks = ["surface", "group", "robot"]
    task_titles = {
        "surface": "Surface",
        "robot": "Robot",
        "group": "Human"
    }
    
    # Create figure with 3 horizontally stacked subplots
    # 3.5 inches wide (single column width), height adjusted for aspect ratio
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 0.9))
    
    for idx, (ax, task) in enumerate(zip(axes, tasks)):
        # Load results
        results = load_results(task)
        chance_level, n_classes = load_dataset_info(task)
        
        if results is None:
            ax.text(0.5, 0.5, f"No data for\n{task} task", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(task_titles[task])
            continue
        
        # Extract data
        seq_lengths = sorted(results.keys())
        test_accs = [results[sl]["final_test_acc"] for sl in seq_lengths]
        
        # Plot test accuracy
        ax.plot(seq_lengths, test_accs, 
               marker='o', linewidth=1.5, markersize=5,
               color='#1f77b4', label='Transformer')
        
        # Plot chance level
        if chance_level is not None:
            ax.axhline(y=chance_level, color='#d62728', linestyle='--', 
                      linewidth=1.0, label=f'Chance')
        
        # Formatting
        ax.set_xlabel('Sequence Length')
        ax.set_xscale('log')
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels([str(sl) for sl in seq_lengths])
        ax.set_ylim([0.20, 1.05])
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.text(0.05, 0.95, task_titles[task], transform=ax.transAxes,
            ha='left', va='top', fontsize=5)
        
        # Only show ylabel and y-tick labels on leftmost plot
        if idx == 0:
            ax.set_ylabel('Accuracy')
        else:
            ax.set_yticklabels([])
        
        # Add legend only to rightmost subplot
        if idx == 2:
            ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=0.3)
    
    # Save figure
    output_path = 'plots/paper_figure_classification_results.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPaper figure saved to: {output_path}")
    
    plt.close()


def print_summary():
    """Print summary statistics for all tasks."""
    tasks = ["surface", "robot", "group"]
    
    print("\n" + "=" * 80)
    print("SUMMARY: Classification Results Across All Tasks")
    print("=" * 80)
    
    for task in tasks:
        results = load_results(task)
        chance_level, n_classes = load_dataset_info(task)
        
        if results is None:
            print(f"\n{task.upper()}: No results found")
            continue
        
        print(f"\n{task.upper()} Classification ({n_classes} classes):")
        print(f"  Chance level accuracy: {chance_level:.4f}")
        print(f"  {'Seq Len':<10} {'Test Acc':<12} {'vs Chance':<12}")
        print(f"  {'-'*34}")
        
        seq_lengths = sorted(results.keys())
        for sl in seq_lengths:
            test_acc = results[sl]["final_test_acc"]
            improvement = test_acc - chance_level if chance_level else 0
            print(f"  {sl:<10} {test_acc:<12.4f} {improvement:+.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Creating paper figure for timeseries classification results...")
    create_paper_figure()
    print_summary()
