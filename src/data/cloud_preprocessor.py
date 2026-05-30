"""Cloud traffic preprocessor for CIC-IDS2018 dataset.

Handles CIC-IDS2018-specific column names and cleaning.
CICFlowMeter output format compatible.
"""

import numpy as np
import pandas as pd

from src.data.base_preprocessor import BasePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


_NON_FEATURE_COLS = {
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "Flow ID",
    "Unnamed: 0",
    "id",
    "ID",
    "index",
}


def _normalize_cloud_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names for CIC-IDS2018 compatibility."""
    return df.rename(columns={c: c.strip() for c in df.columns})


def _coerce_numeric_columns(df: pd.DataFrame, skip_cols: set) -> pd.DataFrame:
    """Convert object-dtype columns to numeric where possible.

    CIC-IDS2018 CSVs are sometimes read entirely as strings due to
    mixed-type rows. This forces numeric conversion so that
    select_dtypes(include=["number"]) returns all feature columns.

    Args:
        df: Input DataFrame.
        skip_cols: Column names to leave as-is (label, identifiers).

    Returns:
        DataFrame with object columns coerced to float where possible.
    """
    for col in df.columns:
        if col in skip_cols:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_ratio = converted.notna().mean()
            if non_null_ratio >= 0.5:
                converted = converted.replace([np.inf, -np.inf], np.nan)
                converted = converted.fillna(converted.median())
                df[col] = converted
                logger.debug("Coerced column '%s' to numeric (%.0f%% valid)", col, non_null_ratio * 100)
    return df


class CloudPreprocessor(BasePreprocessor):
    """Preprocessor for CIC-IDS2018 cloud/traditional network traffic."""

    def __init__(self) -> None:
        super().__init__(domain="cloud")

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Select numeric columns excluding target and flow identifiers."""
        target_col = self._config["target_column"]
        exclude = _NON_FEATURE_COLS | {target_col}
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        cols = [c for c in numeric if c not in exclude]
        logger.debug("Feature columns selected: %s", len(cols))
        return cols

    def _domain_specific_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cloud-specific: strip column names, coerce to numeric, drop constants."""
        target_col = self._config["target_column"]
        df = _normalize_cloud_columns(df)
        df = _coerce_numeric_columns(df, skip_cols=_NON_FEATURE_COLS | {target_col})
        nunique = df.nunique()
        constant = nunique[nunique <= 1].index.tolist()
        # Never drop the target column
        constant = [c for c in constant if c != target_col]
        if constant:
            df = df.drop(columns=constant, errors="ignore")
            logger.debug("Dropped %s constant columns", len(constant))
        return df
