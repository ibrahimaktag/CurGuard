"""Abstract base class for domain-specific preprocessors.

Defines the interface: fit (on train only), transform, fit_transform.
Scaler and encoders are fit ONLY on training data to prevent data leakage.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import get_seed

logger = get_logger(__name__)


class BasePreprocessor(ABC):
    """Abstract preprocessor for network traffic data.

    Subclasses must implement _get_feature_columns and optionally
    _domain_specific_cleanup. Scaler is fit only on train data.
    """

    def __init__(self, domain: str) -> None:
        """Initialize preprocessor for a domain.

        Args:
            domain: 'iot' or 'cloud' (key in datasets.yaml).
        """
        self.domain = domain
        self._config = load_config("datasets")[domain]
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._feature_columns: list[str] | None = None
        self._fitted = False

    @abstractmethod
    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of numeric feature columns (excluding target).

        Args:
            df: Raw DataFrame.

        Returns:
            List of column names to use as features.
        """
        pass

    def _domain_specific_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional domain-specific cleaning. Override in subclass.

        Args:
            df: DataFrame to clean.

        Returns:
            Cleaned DataFrame.
        """
        return df

    def _resolve_target_column(self, df: pd.DataFrame) -> str:
        """Find actual target column name in df, case-insensitively.

        Config may specify 'Label' but CSV may contain 'label' or 'LABEL'.
        When exact match exists it is preferred; falls back to case-insensitive.

        Args:
            df: DataFrame to search.

        Returns:
            Actual column name present in df.

        Raises:
            KeyError: If no case-insensitive match is found.
        """
        target_col = self._config["target_column"]
        if target_col in df.columns:
            return target_col
        lower_map = {c.lower(): c for c in df.columns}
        match = lower_map.get(target_col.lower())
        if match:
            logger.debug(
                "Target column '%s' not found; using case-insensitive match '%s'",
                target_col,
                match,
            )
            return match
        raise KeyError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common cleaning: drop nulls in target, infinities, duplicates."""
        target_col = self._resolve_target_column(df)
        # Rename to the canonical name from config so downstream code is consistent
        canonical = self._config["target_column"]
        if target_col != canonical:
            df = df.rename(columns={target_col: canonical})
        df = df.dropna(subset=[canonical])
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df = df.drop_duplicates(ignore_index=True)
        return self._domain_specific_cleanup(df)

    def fit(self, df: pd.DataFrame) -> "BasePreprocessor":
        """Fit scaler and label encoder on training data only.

        CRITICAL: Call this only on train split. Never on val/test.

        Args:
            df: Training DataFrame.

        Returns:
            self for chaining.
        """
        df = self._clean(df.copy())
        self._feature_columns = self._get_feature_columns(df)
        target_col = self._config["target_column"]

        X = df[self._feature_columns]
        self._scaler = StandardScaler()
        self._scaler.fit(X)

        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(df[target_col].astype(str))

        self._fitted = True
        logger.info(
            "Preprocessor fitted: domain=%s, features=%s, classes=%s",
            self.domain,
            len(self._feature_columns),
            len(self._label_encoder.classes_),
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Transform data using fitted scaler and encoder.

        Args:
            df: DataFrame to transform.
            include_target: If True, return (X, y). Else return X only.

        Returns:
            (X_scaled, y_encoded) or X_scaled only.

        Raises:
            RuntimeError: If fit() was not called.
        """
        if not self._fitted or self._scaler is None or self._label_encoder is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        df = self._clean(df.copy())
        target_col = self._config["target_column"]

        available = [c for c in self._feature_columns if c in df.columns]
        missing = [c for c in self._feature_columns if c not in df.columns]
        if missing:
            logger.warning("Missing feature columns (filled with 0): %s", missing)
            for c in missing:
                df[c] = 0
        X = df[self._feature_columns].values
        X = self._scaler.transform(X)

        if include_target:
            y = self._label_encoder.transform(df[target_col].astype(str))
            return X.astype(np.float32), y.astype(np.int32)
        return X.astype(np.float32)

    def fit_transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Fit on data and transform. Use only for training pipeline.

        Args:
            df: Training DataFrame.
            include_target: If True, return (X, y).

        Returns:
            Transformed arrays.
        """
        self.fit(df)
        return self.transform(df, include_target=include_target)

    @property
    def feature_columns(self) -> list[str]:
        """Return feature column names after fit."""
        if self._feature_columns is None:
            raise RuntimeError("Preprocessor not fitted.")
        return self._feature_columns

    @property
    def n_classes(self) -> int:
        """Return number of label classes after fit."""
        if self._label_encoder is None:
            raise RuntimeError("Preprocessor not fitted.")
        return len(self._label_encoder.classes_)

    @property
    def class_names(self) -> list[str]:
        """Return label class names after fit."""
        if self._label_encoder is None:
            raise RuntimeError("Preprocessor not fitted.")
        return list(self._label_encoder.classes_)
