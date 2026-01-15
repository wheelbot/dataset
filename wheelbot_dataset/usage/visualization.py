import os
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from wheelbot_dataset.usage.dataset import Experiment, ExperimentGroup, Dataset


def plot_timeseries(
    experiments: Union["Experiment", List["Experiment"], "ExperimentGroup"],
    groups: Dict[str, List[str]],
    pdf_path: Optional[str] = None,
    subsample_ms: Optional[float] = None,
    fields_time: str = "_time",
    save_fig: bool = True,
):
    """
    Generate timeseries plots for experiments and save to PDF.
    
    Args:
        experiments: Single Experiment, list of Experiments, or ExperimentGroup.
        groups: Dictionary mapping plot titles to lists of field names to plot together.
        pdf_path: Path for output PDF. If None, uses experiment CSV path with .timeseries.pdf suffix.
        subsample_ms: Optional subsampling interval in milliseconds for plotting.
        fields_time: Name of the time field in the data (default: "_time").
        save_fig: Whether to save the figure to the PDF (default: True).
    Example:
        plot_timeseries(
            exp,
            groups={
                "Angles": ["roll", "pitch", "yaw"],
                "Velocities": ["vx", "vy", "vz"]
            },
            pdf_path="output.pdf"
        )
    """

    # Normalize input to a list of experiments
    if isinstance(experiments, ExperimentGroup):
        experiments = experiments.experiments
    if isinstance(experiments, Experiment):
        experiments = [experiments]

    for exp in experiments:

        df = exp.data.copy().reset_index()
        if fields_time not in df.columns:
            raise ValueError(f"Time field '{fields_time}' not found in data.")

        if subsample_ms is not None:
            original_dt = df[fields_time].iloc[1] - df[fields_time].iloc[0]
            target_dt = subsample_ms / 1000.0
            step = max(1, int(round(target_dt / original_dt)))
            df = df.iloc[::step].reset_index(drop=True)

        # Keep only fields that exist
        groups_filtered = {
            title: [c for c in cols if c in df.columns]
            for title, cols in groups.items()
            if any(c in df.columns for c in cols)
        }

        if pdf_path is None:
            out_path = exp.csv_path.replace(".csv", ".timeseries.pdf")
        else:
            base, ext = os.path.splitext(pdf_path)
            out_path = f"{base}_{os.path.basename(exp.csv_path).replace('.csv','')}.pdf"

        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with PdfPages(out_path) as pdf:

            groups_per_page = 6
            group_list = list(groups_filtered.items())

            for page_start in range(0, len(group_list), groups_per_page):
                chunk = group_list[page_start:page_start + groups_per_page]
                fig_height = 3 * len(chunk)
                fig = plt.figure(figsize=(14, fig_height))

                for i, (title, cols) in enumerate(chunk):
                    ax = fig.add_subplot(len(chunk), 1, i + 1)

                    for col in cols:
                        ax.plot(df[fields_time], df[col], label=col)

                    ax.set_title(title)
                    ax.set_xlabel("time [s]")
                    ax.grid(True)
                    ax.legend(loc="upper right")

                plt.tight_layout()
                if save_fig:
                    pdf.savefig(fig)
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)

        if save_fig:
            print(f"Saved timeseries PDF: {out_path}")
        

def plot_histograms(
    data: Union["Experiment", List["Experiment"], "ExperimentGroup", "Dataset"],
    pdf_path: Optional[str] = None,
    bins: int = 100,
    per_page: int = 12,
    save_fig: bool = True,
):
    """
    Generate histogram plots for all numeric fields and save to PDF.
    
    Args:
        data: Experiment, list of Experiments, ExperimentGroup, or Dataset to analyze.
        pdf_path: Path for output PDF file. If None, uses experiment CSV path with .histograms.pdf suffix.
        bins: Number of bins for histograms (default: 100).
        per_page: Number of histograms per page (default: 12, arranged as 4 rows × 3 columns).
        
    Note:
        All experiments are concatenated before computing histograms.
        Only numeric fields are included in the output.
    """

    # Normalize input
    if isinstance(data, Experiment):
        dfs = [data.data]

    elif isinstance(data, ExperimentGroup):
        dfs = [e.data for e in data.experiments]

    elif isinstance(data, Dataset):
        dfs = []
        for gname, prefixes in data.groups.items():
            group = data.load_group(gname)
            dfs.extend([e.data for e in group.experiments])

    elif isinstance(data, list):
        dfs = [d.data for d in data]

    else:
        raise ValueError("Unsupported data container")

    df = pd.concat(dfs, axis=0, ignore_index=True)
    numeric_df = df.select_dtypes(include=[np.number])
    fields = sorted(numeric_df.columns)

    if pdf_path is None:
        if isinstance(data, Experiment):
            out_path = data.csv_path.replace(".csv", ".histograms.pdf")
        else:
            out_path = "histograms.pdf"
    else:
        out_path = pdf_path

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with PdfPages(out_path) as pdf:

        for start in range(0, len(fields), per_page):
            chunk = fields[start:start + per_page]

            rows = int(np.ceil(len(chunk) / 3))
            fig = plt.figure(figsize=(15, 5 * rows))

            for i, field in enumerate(chunk):
                ax = fig.add_subplot(rows, 3, i + 1)
                values = numeric_df[field].dropna().values
                ax.hist(values, bins=bins)
                ax.set_title(field)
                ax.grid(True)

            plt.tight_layout()
            if save_fig:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

    if save_fig:
        print(f"Saved histogram PDF: {pdf_path}")