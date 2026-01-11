"""
Recording subpackage for wheelbot-dataset.

Contains modules for running experiments and recording data from the wheelbot robot.
"""

from wheelbot_dataset.recording.experiment import (
    run_experiment,
    plot_and_run_sequence,
    plot_and_run_with_repeat,
    RemoteProgramController,
    VideoRecorder,
)

from wheelbot_dataset.recording.prb_sequences import (
    generate_setpoints,
    convert_yaw_setpoints_to_deltas,
    generate_yaw_prbs
)

from wheelbot_dataset.recording.utils import (
    next_log_number,
    continue_skip_abort,
)

from wheelbot_dataset.recording.csvplot import plot_csv_preview

__all__ = [
    "run_experiment",
    "plot_and_run_sequence",
    "plot_and_run_with_repeat",
    "RemoteProgramController",
    "VideoRecorder",
    "generate_setpoints",
    "convert_yaw_setpoints_to_deltas",
    "generate_yaw_prbs",
    "next_log_number",
    "continue_skip_abort",
    "plot_csv_preview",
]
