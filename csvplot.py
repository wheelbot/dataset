import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

def plot_csv_preview(csv_path, subsample_ms=10):
    """
    Load a CSV file, subsample it, plot grouped signals into a
    multi-page PDF with tall shared-x figures.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    subsample_ms : float
        Desired sampling interval in milliseconds. Default = 10ms.
        Assumes input data has 1ms resolution.
    """

    # ----------------------------------------------------------
    # Load and subsample
    # ----------------------------------------------------------
    df = pd.read_csv(csv_path)

    # assume `_time` is in seconds
    # original sampling ~1ms → 1000 Hz
    original_dt = (df["_time"].iloc[1] - df["_time"].iloc[0])
    target_dt = subsample_ms / 1000.0

    step = max(1, int(round(target_dt / original_dt)))
    df = df.iloc[::step].reset_index(drop=True)

    # ----------------------------------------------------------
    # Define groups to plot together
    # (You can customize this dictionary)
    # ----------------------------------------------------------
    groups = {
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

    # Keep only columns that actually exist in CSV
    groups = {
        title: [c for c in cols if c in df.columns]
        for title, cols in groups.items()
        if any(c in df.columns for c in cols)
    }

    # ----------------------------------------------------------
    # Create PDF
    # ----------------------------------------------------------
    pdf_path = csv_path.replace(".csv", ".preview.pdf")
    with PdfPages(pdf_path) as pdf:

        # We can put ~6–8 groups per page depending on preference
        groups_per_page = 6
        group_list = list(groups.items())

        for page_start in range(0, len(group_list), groups_per_page):
            chunk = group_list[page_start:page_start + groups_per_page]

            # Figure height = 3 inch per subplot
            fig_height = 3 * len(chunk)
            fig = plt.figure(figsize=(14, fig_height))

            for i, (title, cols) in enumerate(chunk):
                ax = fig.add_subplot(len(chunk), 1, i + 1)

                for col in cols:
                    ax.plot(df["_time"], df[col], label=col)

                ax.set_title(title)
                ax.set_xlabel("time [s]")
                ax.legend(loc="upper right")
                ax.grid(True)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path