"""Validate `configs/router.yaml` feature_mapping against on-disk CSV headers.

Used before training the router to catch renamed or missing columns early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RouterMappingValidationResult:
    """Outcome of comparing router.yaml mapping to dataset column names."""

    missing_iot: list[str] = field(default_factory=list)
    missing_cloud: list[str] = field(default_factory=list)
    iot_sample: Path | None = None
    cloud_sample: Path | None = None
    iot_columns: int = 0
    cloud_columns: int = 0

    @property
    def ok(self) -> bool:
        """True if every mapped column exists in both domains."""
        return not self.missing_iot and not self.missing_cloud


def read_csv_column_names(path: Path) -> set[str]:
    """Read only the header row and return stripped column names.

    Args:
        path: Path to a CSV file.

    Returns:
        Set of column name strings.
    """
    df = pd.read_csv(path, nrows=0, low_memory=False)
    return {str(c).strip() for c in df.columns}


def validate_mapping_vs_columns(
    feature_mapping: dict[str, Any],
    iot_columns: set[str],
    cloud_columns: set[str],
) -> RouterMappingValidationResult:
    """Check that each mapped IoT/Cloud source column exists in the given sets.

    Args:
        feature_mapping: The `feature_mapping` block from router.yaml.
        iot_columns: Column names present in an IoT CSV.
        cloud_columns: Column names present in a Cloud CSV.

    Returns:
        RouterMappingValidationResult with per-domain missing entries.
    """
    result = RouterMappingValidationResult()
    for common_name, aliases in feature_mapping.items():
        if not isinstance(aliases, dict):
            continue
        iot_col = aliases.get("iot")
        cloud_col = aliases.get("cloud")
        if iot_col and str(iot_col).strip() not in iot_columns:
            result.missing_iot.append(f"{common_name} → '{iot_col}'")
        if cloud_col and str(cloud_col).strip() not in cloud_columns:
            result.missing_cloud.append(f"{common_name} → '{cloud_col}'")
    result.iot_columns = len(iot_columns)
    result.cloud_columns = len(cloud_columns)
    return result


def _first_csv(path: Path) -> Path | None:
    """Return the first *.csv path in a directory, sorted by name."""
    if not path.is_dir():
        return None
    files = sorted(path.glob("*.csv"))
    return files[0] if files else None


def validate_router_mapping_from_disk() -> RouterMappingValidationResult:
    """Load router + datasets config, read one CSV header per domain, validate.

    Returns:
        RouterMappingValidationResult. If raw folders are empty, missing lists
        stay empty and sample paths stay None (caller should treat as error).

    Raises:
        FileNotFoundError: If configured raw directories do not exist.
    """
    router_cfg = load_config("router")
    mapping = router_cfg.get("feature_mapping", {})
    ds = load_config("datasets")
    root = get_project_root()

    result = RouterMappingValidationResult()
    iot_dir = root / ds["iot"]["raw_path"]
    cloud_dir = root / ds["cloud"]["raw_path"]

    if not iot_dir.is_dir():
        raise FileNotFoundError(f"IoT raw path not found: {iot_dir}")
    if not cloud_dir.is_dir():
        raise FileNotFoundError(f"Cloud raw path not found: {cloud_dir}")

    iot_csv = _first_csv(iot_dir)
    cloud_csv = _first_csv(cloud_dir)

    if iot_csv is None:
        logger.warning("No CSV files in %s", iot_dir)
        return result
    if cloud_csv is None:
        logger.warning("No CSV files in %s", cloud_dir)
        return result

    result.iot_sample = iot_csv
    result.cloud_sample = cloud_csv

    iot_cols = read_csv_column_names(iot_csv)
    cloud_cols = read_csv_column_names(cloud_csv)

    checked = validate_mapping_vs_columns(mapping, iot_cols, cloud_cols)
    result.missing_iot = checked.missing_iot
    result.missing_cloud = checked.missing_cloud
    result.iot_columns = checked.iot_columns
    result.cloud_columns = checked.cloud_columns
    return result
