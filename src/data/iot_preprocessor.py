"""IoT traffic preprocessor for CIC-IoT2023 dataset.

Handles CIC-IoT2023-specific column names and cleaning.
Feature columns are inferred from numeric columns excluding target.
"""

from typing import Any

import pandas as pd

from src.data.base_preprocessor import BasePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

def _normalize_iot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for CIC-IoT2023 compatibility."""
    cols = {c: c.strip() for c in df.columns}
    return df.rename(columns=cols)


class IoTPreprocessor(BasePreprocessor):
    """Preprocessor for CIC-IoT2023 IoT traffic data."""

    def __init__(self) -> None:
        super().__init__(domain="iot")

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Select numeric columns excluding target and non-feature columns."""
        target_col = self._config["target_column"]
        exclude = {target_col, "id", "ID", "index", "Unnamed: 0"}
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in numeric if c not in exclude]

    def _domain_specific_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """IoT-specific: drop constant columns, handle protocol encoding."""
        df = _normalize_iot_columns(df)
        nunique = df.nunique()
        constant = nunique[nunique <= 1].index.tolist()
        if constant:
            df = df.drop(columns=[c for c in constant if c in df.columns], errors="ignore")
            logger.debug("Dropped constant columns: %s", constant)
        return df
