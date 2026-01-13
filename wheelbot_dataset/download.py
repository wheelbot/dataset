#!/usr/bin/env python3
"""
Dataset download utilities for wheelbot-dataset.

This module provides functionality to automatically download the wheelbot
dataset from Zenodo if it's not already available locally.
"""

import os
import requests
import zipfile
from pathlib import Path
from typing import Optional
import fire


# Default Zenodo record ID (update this when the dataset is published)
DEFAULT_ZENODO_RECORD_ID = None  # Will be set once dataset is published on Zenodo

# Default dataset directory name
DEFAULT_DATASET_DIR = "data"


def download_dataset(
    output_dir: str = DEFAULT_DATASET_DIR,
    zenodo_record_id: Optional[int] = None,
    force: bool = False
):
    """
    Download the wheelbot dataset from Zenodo.
    
    This function checks if the dataset already exists locally. If not, it downloads
    the dataset as a zip file from Zenodo and extracts it to the specified directory.
    
    Args:
        output_dir: Directory where the dataset should be stored (default: "data").
                   If the directory already contains data and force=False, download is skipped.
        zenodo_record_id: Zenodo record ID for the dataset. If not provided, uses the
                         default record ID configured in this module.
        force: If True, download and extract even if the dataset directory already exists.
               Warning: This will delete the existing directory first.
    
    Returns:
        None
        
    Raises:
        ValueError: If no Zenodo record ID is provided and no default is configured.
        requests.HTTPError: If download from Zenodo fails.
        
    Example:
        >>> download_dataset()  # Downloads to ./data
        >>> download_dataset(output_dir="my_data", zenodo_record_id=17081411)
    """
    # Use default record ID if none provided
    if zenodo_record_id is None:
        zenodo_record_id = DEFAULT_ZENODO_RECORD_ID
    
    if zenodo_record_id is None:
        raise ValueError(
            "No Zenodo record ID provided. Either pass zenodo_record_id parameter "
            "or update DEFAULT_ZENODO_RECORD_ID in wheelbot_dataset/download.py"
        )
    
    output_path = Path(output_dir)
    
    # Check if dataset already exists
    if output_path.exists() and not force:
        # Check if directory has content (likely already downloaded)
        if list(output_path.iterdir()):
            print(f"Dataset directory '{output_dir}' already exists and contains files.")
            print("Skipping download. Use --force to re-download.")
            return
    
    # If force is True and directory exists, remove it
    if force and output_path.exists():
        print(f"Force mode: Removing existing directory '{output_dir}'...")
        import shutil
        shutil.rmtree(output_path)
    
    # Create a temporary directory for the download
    temp_dir = Path(f".wheelbot_dataset_download_temp")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Setup session with appropriate headers
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) WheelbotDatasetDownloader/1.0 (+https://zenodo.org/)",
            "Accept": "application/json",
        })
        
        # Get record information from Zenodo API
        api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
        print(f"Fetching dataset information from Zenodo (record {zenodo_record_id})...")
        
        resp = session.get(api_url, timeout=60)
        
        if resp.status_code == 403:
            print("Got 403 from Zenodo API.")
            print("Response headers:", resp.headers)
            print("Response body (first 1000 chars):")
            print(resp.text[:1000])
            resp.raise_for_status()
        
        resp.raise_for_status()
        record = resp.json()
        
        # Find the data.zip file
        files = record.get("files", [])
        if not files:
            raise ValueError("No files found in Zenodo record.")
        
        data_zip_file = None
        for f in files:
            if f["key"] == "data.zip":
                data_zip_file = f
                break
        
        if data_zip_file is None:
            # List available files
            available_files = [f["key"] for f in files]
            raise ValueError(
                f"'data.zip' not found in Zenodo record. "
                f"Available files: {', '.join(available_files)}"
            )
        
        # Download the zip file
        filename = data_zip_file["key"]
        download_url = data_zip_file["links"]["self"]
        file_size = data_zip_file.get("size", 0)
        zip_path = temp_dir / filename
        
        print(f"Downloading {filename} ({file_size / (1024**3):.2f} GB)...")
        print(f"This may take a while depending on your connection speed...")
        
        with session.get(download_url, stream=True, timeout=300) as r:
            if r.status_code == 403:
                print(f"403 while downloading {filename}")
                print("Body (first 500 chars):", r.text[:500])
                r.raise_for_status()
            
            r.raise_for_status()
            
            # Download with progress indication
            downloaded = 0
            with open(zip_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        # Print progress every 100MB
                        if downloaded % (100 * 1024 * 1024) == 0 or downloaded == file_size:
                            progress = (downloaded / file_size * 100) if file_size > 0 else 0
                            print(f"  Downloaded: {downloaded / (1024**3):.2f} GB / {file_size / (1024**3):.2f} GB ({progress:.1f}%)")
        
        print(f"Download complete!")
        
        # Extract the zip file
        print(f"Extracting dataset to '{output_dir}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        print(f"Extraction complete!")
        
        # Verify the output directory exists
        if not output_path.exists():
            raise RuntimeError(
                f"Expected directory '{output_dir}' not found after extraction. "
                "The zip file structure may be different than expected."
            )
        
        print(f"\nDataset successfully downloaded and extracted to: {output_path.absolute()}")
        
    except Exception as e:
        print(f"\nError during download: {e}")
        raise
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files.")


def check_dataset(dataset_dir: str = DEFAULT_DATASET_DIR):
    """
    Check if the dataset exists locally and print information about it.
    
    Args:
        dataset_dir: Directory where the dataset should be located.
    
    Returns:
        None: Prints dataset status to stdout.
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        print("Run 'python -m wheelbot_dataset download' to download the dataset.")
        return
    
    if not list(dataset_path.iterdir()):
        print(f"Dataset directory '{dataset_dir}' exists but is empty.")
        print("Run 'python -m wheelbot_dataset download' to download the dataset.")
        return
    
    # Try to load the dataset to get statistics
    try:
        from wheelbot_dataset.usage.dataset import Dataset
        ds = Dataset(str(dataset_path))
        
        print(f"Dataset found at: {dataset_path.absolute()}")
        print(f"\nDataset groups:")
        for group_name in sorted(ds.groups.keys()):
            num_experiments = len(ds.groups[group_name].experiments)
            print(f"  {group_name}: {num_experiments} experiments")
        
        total_experiments = sum(len(group.experiments) for group in ds.groups.values())
        print(f"\nTotal experiments: {total_experiments}")
        
    except Exception as e:
        print(f"Dataset directory exists at: {dataset_path.absolute()}")
        print(f"But could not load dataset: {e}")


if __name__ == "__main__":
    fire.Fire({
        "download": download_dataset,
        "check": check_dataset,
    })
