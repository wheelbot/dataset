import os
import glob
import json
import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any, List
from typing import Dict, List, Union, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle


def default_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Identity filter; user can override with any acausal/causal filter."""
    return df


class Experiment:
    """
    Represents a single experiment with timeseries data and metadata.
    
    Loads CSV data and JSON metadata on initialization. Provides methods for
    filtering, resampling, time-based cutting, and data export.
    """
    
    def __init__(self, csv_path: str, meta_path: str):
        """
        Initialize an experiment by loading data and metadata.
        
        Args:
            csv_path: Path to the CSV file containing timeseries data.
            meta_path: Path to the JSON file containing experiment metadata.
        """
        self.csv_path = csv_path
        self.meta_path = meta_path
        self.data = self._load_csv(csv_path)
        self.meta = self._load_meta(meta_path)

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        """
        Load CSV file and set time column as index if present.
        
        Args:
            path: Path to the CSV file.
            
        Returns:
            DataFrame with '_time' column as index if it exists.
        """
        df = pd.read_csv(path)
        if "_time" in df.columns:
            df = df.set_index("_time")
        return df

    @staticmethod
    def _load_meta(path: str) -> Dict[str, Any]:
        """
        Load experiment metadata from JSON file.
        
        Args:
            path: Path to the JSON metadata file.
            
        Returns:
            Dictionary containing metadata.
        """
        with open(path, "r") as f:
            return json.load(f)

    def apply_filter(self, filter_fn: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        Apply a filter function to the experiment data.
        
        Args:
            filter_fn: Function that takes a DataFrame and returns a filtered DataFrame.
                      Can be acausal (e.g., butterworth filtfilt) or causal.
                      
        Returns:
            New Experiment instance with filtered data.
        """
        new = Experiment.__new__(Experiment)
        new.csv_path = self.csv_path
        new.meta_path = self.meta_path
        new.meta = self.meta
        new.data = filter_fn(self.data.copy(deep=True))
        return new

    def resample(self, dt: float):
        """
        Resample timeseries data to a new time step.
        
        Args:
            dt: New time step in seconds.
            
        Returns:
            New Experiment instance with resampled data using nearest neighbor interpolation.
            
        Note:
            Assumes the DataFrame index represents time in seconds.
        """
        new_index = np.arange(self.data.index[0],
                              self.data.index[-1],
                              dt)
        df_resampled = self.data.reindex(
            new_index,
            method="nearest"
        )
        new = Experiment.__new__(Experiment)
        new.csv_path = self.csv_path
        new.meta_path = self.meta_path
        new.meta = self.meta
        new.data = df_resampled
        return new


    def cut_time(self, start: float = 0.0, end: float = 0.0):
        """
        Trim data by removing time from the beginning and end.
        
        Args:
            start: Number of seconds to remove from the beginning.
            end: Number of seconds to remove from the end.
            
        Returns:
            New Experiment instance with trimmed data.
        """
        t0 = self.data.index[0] + start
        t1 = self.data.index[-1] - end
        df = self.data[(self.data.index >= t0) & (self.data.index <= t1)]

        new = Experiment.__new__(Experiment)
        new.csv_path = self.csv_path
        new.meta_path = self.meta_path
        new.meta = self.meta
        new.data = df
        return new

    def cut_by_condition(
        self,
        start_condition: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        end_condition: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    ):
        """
        Trim data based on boolean conditions on the data.
        
        Args:
            start_condition: Function that takes a DataFrame and returns a boolean Series.
                           Data before the first True value is removed.
            end_condition: Function that takes a DataFrame and returns a boolean Series.
                          Data is kept until all remaining samples satisfy the condition.
                          
        Returns:
            New Experiment instance with conditionally trimmed data.
            
        Example:
            exp.cut_by_condition(
                start_condition=lambda df: df['torque'].abs() > 0.1,
                end_condition=lambda df: df['velocity'].abs() < 0.01
            )
        """
        df = self.data

        if start_condition is not None:
            cond = start_condition(df)
            if cond.any():
                t_start = df.index[cond.argmax()]
                df = df[df.index >= t_start]

        if end_condition is not None:
            cond = end_condition(df)
            end_idx = None
            for i in range(len(df)):
                if cond.iloc[i:].all():
                    end_idx = i
                    break
            if end_idx is not None:
                t_end = df.index[end_idx]
                df = df[df.index <= t_end]

        new = Experiment.__new__(Experiment)
        new.csv_path = self.csv_path
        new.meta_path = self.meta_path
        new.meta = self.meta
        new.data = df
        return new

    def filter_by_metadata(self, **conditions):
        """
        Filter experiment based on metadata conditions.
        
        Args:
            **conditions: Key-value pairs that must match in the metadata.
            
        Returns:
            Self if all conditions are satisfied, None otherwise.
            
        Example:
            exp.filter_by_metadata(experiment_status="success", robot="wheelbot-beta-2")
        """
        ok = all(self.meta.get(k) == v for k, v in conditions.items())
        return self if ok else None

    def to_numpy(self, fields):
        """
        Export specific fields to a numpy array.
        
        Args:
            fields: List of field names to export. Can include 'time' or '_time'
                   to export the time index as a column.
                   
        Returns:
            Numpy array of shape (T, D) where T is the number of timesteps and
            D is the number of fields.
            
        Raises:
            KeyError: If any requested field is not found in the data.
        """
        df = self.data
        
        time_fields = ['_time', 'time']
        regular_fields = []
        arrays = []
        
        for f in fields:
            if f in time_fields:
                arrays.append(df.index.to_numpy().reshape(-1, 1))
            else:
                regular_fields.append(f)
        
        missing = [f for f in regular_fields if f not in df.columns]
        if missing:
            raise KeyError(f"Fields not found in experiment: {missing}")
        
        if regular_fields:
            arrays.append(df[regular_fields].to_numpy())
        
        return np.concatenate(arrays, axis=1) if arrays else np.empty((len(df), 0))

class ExperimentGroup:
    """
    Container for multiple experiments of the same type.
    
    Provides methods to apply operations across all experiments in the group,
    such as filtering, mapping functions, and data export.
    """
    
    def __init__(self, name: str, experiments: List[Experiment]):
        """
        Initialize an experiment group.
        
        Args:
            name: Name of the experiment group.
            experiments: List of Experiment instances.
        """
        self.name = name
        self.experiments = experiments

    def __getitem__(self, idx):
        """
        Get experiment by index.
        
        Args:
            idx: Index of the experiment.
            
        Returns:
            Experiment at the specified index.
        """
        return self.experiments[idx]

    def __len__(self):
        """
        Get number of experiments in the group.
        
        Returns:
            Number of experiments.
        """
        return len(self.experiments)

    def map(self, fn):
        """
        Apply a function to all experiments in the group.
        
        Args:
            fn: Function that takes an Experiment and returns an Experiment or None.
                Experiments returning None are excluded from the result.
                
        Returns:
            New ExperimentGroup with processed experiments, or a list of numpy arrays
            if all processed results are arrays.
        """
        processed = []
        for exp in self.experiments:
            new_exp = fn(exp)
            if new_exp is not None:
                processed.append(new_exp)
        if processed and all(isinstance(exp, np.ndarray) for exp in processed):
            return processed
        return ExperimentGroup(self.name, processed)

    def to_numpy(self, fields):
        """
        Export all experiments to a single concatenated numpy array.
        
        Args:
            fields: List of field names to export from each experiment.
            
        Returns:
            Numpy array of shape (T_total, D) where T_total is the sum of all
            timesteps across experiments and D is the number of fields.
        """
        arrays = [exp.to_numpy(fields) for exp in self.experiments]
        return np.concatenate(arrays, axis=0)

class Dataset:
    """
    Container for multiple experiment groups organized by directory structure.
    
    Automatically discovers and loads experiment groups from subdirectories in the
    root folder. Each subdirectory becomes an ExperimentGroup containing all
    experiments (CSV + metadata pairs) found within.
    """
    
    def __init__(self, root: str):
        """
        Load dataset from a root directory.
        
        Args:
            root: Path to the root directory containing experiment group folders.
                 Each subdirectory should contain CSV files with matching .meta files.
                 
        Attributes:
            groups: Dictionary mapping group names to ExperimentGroup instances.
        """
        self.root = root
        self.groups = self._load_groups()

    def _load_groups(self) -> Dict[str, ExperimentGroup]:
        """
        Discover and load all experiment groups from subdirectories.
        
        Returns:
            Dictionary mapping group directory names to ExperimentGroup instances.
            Only includes groups with at least one valid experiment.
        """
        groups: Dict[str, ExperimentGroup] = {}

        for group_dir in sorted(os.listdir(self.root)):
            full_path = os.path.join(self.root, group_dir)
            if not os.path.isdir(full_path):
                continue

            csvs = sorted(glob.glob(os.path.join(full_path, "*.csv")))
            experiments: List[Experiment] = []

            for csv_path in csvs:
                prefix, _ = os.path.splitext(csv_path)
                meta_path = prefix + ".meta"
                if os.path.exists(meta_path):
                    experiments.append(Experiment(csv_path, meta_path))

            if experiments:
                groups[group_dir] = ExperimentGroup(group_dir, experiments)

        return groups

    def load_group(self, group_name: str) -> ExperimentGroup:
        """
        Get an experiment group by name.
        
        Args:
            group_name: Name of the group (subdirectory name).
            
        Returns:
            ExperimentGroup instance.
            
        Raises:
            KeyError: If the group name is not found.
        """
        return self.groups[group_name]
    
    def map(self, fn):
        """
        Apply a function to all experiments across all groups in the dataset.
        
        Args:
            fn: Function that takes an Experiment and returns an Experiment or None.
                Experiments returning None are excluded from the result.
                
        Returns:
            New Dataset instance with processed groups. Empty groups are removed.
        """
        new_ds = Dataset.__new__(Dataset)
        new_ds.root = self.root
        new_ds.groups = {}

        for name, group in self.groups.items():
            processed_group = group.map(fn)
            if len(processed_group) > 0:
                new_ds.groups[name] = processed_group

        return new_ds

def plot_timeseries(
    experiments: Union["Experiment", List["Experiment"], "ExperimentGroup"],
    groups: Dict[str, List[str]],
    pdf_path: Optional[str] = None,
    subsample_ms: Optional[float] = None,
    fields_time: str = "_time",
):
    """
    Generate timeseries plots for experiments and save to PDF.
    
    Args:
        experiments: Single Experiment, list of Experiments, or ExperimentGroup.
        groups: Dictionary mapping plot titles to lists of field names to plot together.
        pdf_path: Path for output PDF. If None, uses experiment CSV path with .timeseries.pdf suffix.
        subsample_ms: Optional subsampling interval in milliseconds for plotting.
        fields_time: Name of the time field in the data (default: "_time").
        
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
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved timeseries PDF: {out_path}")
        

def plot_histograms(
    data: Union["Experiment", List["Experiment"], "ExperimentGroup", "Dataset"],
    pdf_path: str,
    bins: int = 100,
    per_page: int = 12
):
    """
    Generate histogram plots for all numeric fields and save to PDF.
    
    Args:
        data: Experiment, list of Experiments, ExperimentGroup, or Dataset to analyze.
        pdf_path: Path for output PDF file.
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

    with PdfPages(pdf_path) as pdf:

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
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved histogram PDF: {pdf_path}")



def to_prediction_dataset(
    ds: Dataset | ExperimentGroup | Experiment,
    fields_states: List[str],
    fields_actions: List[str],
    fields_observations: List[str] = [],
    N_future: int = 1,
    N_past: int=1,
    skip_N: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert experiments to trajectory snippet prediction dataset format.
    
    Creates overlapping trajectory snippets from time-series data for multi-step prediction.
    Each snippet contains N_past historical states, N_past+N_future-1 actions, and N_future
    future states. The skip_N parameter controls the stride between consecutive snippets.
    
    Args:
        ds: Dataset, ExperimentGroup, or single Experiment to convert.
        fields_states: List of field names representing the state variables.
        fields_actions: List of field names representing the action variables.
        fields_observations: List of field names representing the observations variables.
        N_future: Number of future timesteps to predict (default: 1).
        N_past: Number of past timesteps to include as context (default: 1).
        skip_N: Stride between consecutive snippets in timesteps (default: 1).
                When skip_N < N_future+N_past, snippets will overlap.
        
    Returns:
        Tuple of (states, actions, next_states) as numpy arrays where:
        - states: Historical states (shape: [D, N_past, len(fields_states)])
        - actions: Action sequence (shape: [D, N_past+N_future-1, len(fields_actions)])
        - next_states: Future states to predict (shape: [D, N_future, len(fields_states)])
        
        where D is the total number of trajectory snippets across all experiments.
    """
    
    def exp_to_N_step_prediction(exp: Experiment):
        exp_states_np = exp.to_numpy(fields_states)
        exp_actions_np = exp.to_numpy(fields_actions)
        if fields_observations:
            exp_observations_np = exp.to_numpy(fields_observations)
        num_sections = (exp_actions_np.shape[0]-N_future-N_past)//skip_N
        states = []
        next_states = []
        actions = []
        observations = []
        for i in range(num_sections):
            start = i*skip_N
            states.append(exp_states_np[start:start+N_past])
            if fields_observations:
                observations.append(exp_observations_np[start:start+N_past])
            next_states.append(exp_states_np[start+N_past:start+N_past+N_future])
            actions.append(exp_actions_np[start:start+N_past+N_future-1])
        
        return np.array(states), np.array(actions), np.array(next_states), np.array(observations)
    
    if isinstance(ds, Experiment):
        states, actions, next_states = exp_to_N_step_prediction(ds)
        return states, actions, next_states
    elif isinstance(ds, ExperimentGroup):
        all_states, all_actions, all_next_states, all_observations = [], [], [], []
        for exp in ds.experiments:
            states, actions, next_states, observation = exp_to_N_step_prediction(exp)
            all_states.append(states)
            all_actions.append(actions)
            all_next_states.append(next_states)
            if fields_observations:
                all_observations.append(observation)
        return (
            np.concatenate(all_states, axis=0),
            np.concatenate(all_actions, axis=0),
            np.concatenate(all_next_states, axis=0),
            np.concatenate(all_observations, axis=0) if all_observations else np.array([])
        )
    elif isinstance(ds, Dataset):
        all_states, all_actions, all_next_states, all_observations = [], [], [], []
        for group in ds.groups.values():
            for exp in group.experiments:
                states, actions, next_states, observation = exp_to_N_step_prediction(exp)
                all_states.append(states)
                all_actions.append(actions)
                all_next_states.append(next_states)
                if fields_observations:
                    all_observations.append(observation)
        return (
            np.concatenate(all_states, axis=0),
            np.concatenate(all_actions, axis=0),
            np.concatenate(all_next_states, axis=0),
            np.concatenate(all_observations, axis=0) if all_observations else np.array([])
        )
    raise ValueError("Input must be Experiment, ExperimentGroup, or Dataset")
