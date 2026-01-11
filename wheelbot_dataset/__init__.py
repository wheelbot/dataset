"""
wheelbot-dataset: A package for recording and using wheelbot experiment data.

This package provides tools for:
- Recording experiments with the wheelbot robot (wheelbot_dataset.recording)
- Loading and processing experiment datasets (wheelbot_dataset.usage)
"""

from wheelbot_dataset.usage import (
    Dataset,
    Experiment,
    ExperimentGroup,
    default_filter,
    plot_timeseries,
    plot_histograms,
    to_prediction_dataset,
)

from wheelbot_dataset.recording import (
    run_experiment,
    plot_and_run_sequence,
    plot_and_run_with_repeat,
    RemoteProgramController,
    VideoRecorder,
    generate_setpoints,
    next_log_number,
    continue_skip_abort,
    plot_csv_preview,
)

__all__ = [
    # Usage
    "Dataset",
    "Experiment",
    "ExperimentGroup",
    "default_filter",
    "plot_timeseries",
    "plot_histograms",
    "to_prediction_dataset",
    # Recording
    "run_experiment",
    "plot_and_run_sequence",
    "plot_and_run_with_repeat",
    "RemoteProgramController",
    "VideoRecorder",
    "generate_setpoints",
    "next_log_number",
    "continue_skip_abort",
    "plot_csv_preview",
]
