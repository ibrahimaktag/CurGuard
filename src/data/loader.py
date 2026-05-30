"""CSV-based network traffic data loader.

Loads raw CSV files from dataset directories, validates format,
and handles missing/null values. Paths and column names come from config.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(
    path: Path,
    required_columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a single CSV file with validation.

    Args:
        path: Path to CSV file.
        required_columns: Columns that must exist. If None, only target is checked.
        nrows: Limit rows for testing (None = load all).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception as e:
        logger.error("Failed to load CSV %s: %s", path, e)
        raise

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Null values in %s: %s", path.name, null_counts[null_counts > 0].to_dict())

    return df


def _label_from_filename(path: Path) -> str:
    """Derive class label from CSV filename.

    Strips known suffixes (.csv, .pcap), then removes any trailing
    numeric suffix so that split files merge into a single class.

    Examples:
        Backdoor_Malware.pcap.csv  → "Backdoor_Malware"
        DDoS-ICMP_Flood1.csv       → "DDoS-ICMP_Flood"
        DDoS-ICMP_Flood2.pcap.csv  → "DDoS-ICMP_Flood"
        DDoS-ICMP_Flood.csv        → "DDoS-ICMP_Flood"

    Args:
        path: Path to the CSV file.

    Returns:
        Normalized label string.
    """
    name = path.name
    for suffix in (".csv", ".pcap"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Trailing digits (e.g. "Flood1", "Flood12") → "Flood"
    name = re.sub(r"\d+$", "", name)
    return name


def load_dataset_raw(
    domain: str,
    filename: Optional[str] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw dataset CSV(s) for a domain (iot or cloud).

    Reads paths from configs/datasets.yaml. If filename is None,
    loads all CSVs found in the raw directory.

    For domains with label_source: "filename" (e.g. CIC-IoT2023), each CSV
    does not contain a label column — the label is derived from the filename
    and added as a new column matching target_column.

    Args:
        domain: 'iot' or 'cloud'.
        filename: Specific file to load (e.g. 'Backdoor_Malware.pcap.csv').
            If None, all *.csv files in the raw directory are loaded.
        nrows: Row limit per file for testing (None = load all).

    Returns:
        DataFrame with raw traffic data and target_column populated.

    Raises:
        ValueError: If domain is invalid or no files found.
        FileNotFoundError: If raw path or specific file does not exist.
    """
    config = load_config("datasets")
    if domain not in config:
        raise ValueError(f"Unknown domain: {domain}. Use 'iot' or 'cloud'.")

    cfg = config[domain]
    root = get_project_root()
    raw_path = root / cfg["raw_path"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_path}")

    if filename:
        file_path = raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        paths = [file_path]
    else:
        paths = sorted(raw_path.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No CSV files in {raw_path}")

    target_col = cfg["target_column"]
    label_source = cfg.get("label_source", "column")
    use_filename_label = label_source == "filename"

    dfs = []
    for p in paths:
        if use_filename_label:
            df = load_csv(p, required_columns=None, nrows=nrows)
            label = _label_from_filename(p)
            df[target_col] = label
        else:
            df = load_csv(p, required_columns=[target_col], nrows=nrows)
            # Drop rows where target column contains the column name itself
            # (happens when a CSV header row is accidentally read as data)
            col_actual = next(
                (c for c in df.columns if c.lower() == target_col.lower()), target_col
            )
            mask = df[col_actual].astype(str).str.strip().str.lower() == col_actual.lower()
            if mask.any():
                logger.warning(
                    "Dropped %s header-as-data row(s) in %s", mask.sum(), p.name
                )
                df = df[~mask].reset_index(drop=True)

        dfs.append(df)
        logger.info("Loaded %s rows from %s (label=%s)", len(df), p.name,
                    _label_from_filename(p) if use_filename_label else "column")

    result = pd.concat(dfs, ignore_index=True)
    logger.info("Total loaded: %s rows for domain=%s", len(result), domain)
    return result


def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    """Validate that DataFrame has required structure for preprocessing.

    Args:
        df: DataFrame to validate.
        target_column: Name of label column.

    Raises:
        ValueError: If validation fails.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)}")

    if df[target_column].isnull().any():
        raise ValueError(f"Target column '{target_column}' contains null values")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")


class NetworkDataset(Dataset):
    """
    Custom PyTorch Dataset for network traffic data.
    Converts preprocessed pandas DataFrames into PyTorch Tensors.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str, task_type: str = 'binary'):
        """
        Args:
            df (pd.DataFrame): Preprocessed DataFrame.
            target_col (str): Name of the target label column.
            task_type (str): The classification task type ('binary' or 'multiclass').
        """
        self.target_col = target_col
        self.task_type = task_type
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
            
        # Pre-allocate and convert entire dataset to tensors for CPU efficiency during iteration
        x_array = df.drop(columns=[target_col]).values
        y_array = df[target_col].values
        
        # Features are always float32 for neural network stability
        self.X_tensor = torch.tensor(x_array, dtype=torch.float32)
        
        # Target type depends on the task
        if self.task_type == 'binary':
            # Binary classification typically uses BCEWithLogitsLoss which expects float32
            self.y_tensor = torch.tensor(y_array, dtype=torch.float32)
        elif self.task_type == 'multiclass':
            # Multiclass classification uses CrossEntropyLoss which expects long (int64)
            self.y_tensor = torch.tensor(y_array, dtype=torch.long)
        else:
            raise ValueError("task_type must be 'binary' or 'multiclass'")
        
    def __len__(self) -> int:
        return len(self.X_tensor)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample (features, target) as PyTorch Tensors.
        """
        return self.X_tensor[idx], self.y_tensor[idx]


def create_dataloader(
    df: pd.DataFrame, 
    target_col: str, 
    task_type: str = 'binary',
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Wraps the NetworkDataset in a PyTorch DataLoader for memory-efficient batching.
    Can natively accept cleaned DataFrames from CloudPreprocessor or IoTPreprocessor.
    
    Args:
        df: Preprocessed pandas DataFrame.
        target_col: Name of the target label column.
        task_type: 'binary' or 'multiclass'.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data every epoch.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory.
        
    Returns:
        Configured PyTorch DataLoader instance.
    """
    logger.info(f"PyTorch DataLoader oluşturuluyor (batch_size={batch_size}, task={task_type})...")
    
    dataset = NetworkDataset(df=df, target_col=target_col, task_type=task_type)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info("DataLoader başarıyla oluşturuldu.")
    return loader
