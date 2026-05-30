"""Stratified train / validation / test splits for tabular traffic DataFrames."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _stratify_series(y: pd.Series) -> pd.Series | None:
    """Return y for stratify= if every class has at least two samples, else None."""
    vc = y.value_counts()
    if len(vc) < 2:
        return None
    if vc.min() < 2:
        logger.warning("Stratify disabled: at least one class has < 2 samples.")
        return None
    return y


def split_train_val_test_df(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labeled DataFrame into train, validation, and test sets.

    ``test_size`` and ``val_size`` are fractions of the full dataset (e.g. 0.2 and 0.1).

    Args:
        df: Input DataFrame including the label column.
        target_column: Name of the label column.
        test_size: Fraction for the held-out test set.
        val_size: Fraction for validation (from the remainder after removing test).
        random_state: RNG seed for reproducibility.

    Returns:
        (df_train, df_val, df_test).
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be in (0, 1).")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    y = df[target_column].astype(str)
    strat = _stratify_series(y)

    df_tr_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    y_mid = df_tr_val[target_column].astype(str)
    strat_mid = _stratify_series(y_mid)
    val_fraction_of_train = val_size / (1.0 - test_size)

    df_train, df_val = train_test_split(
        df_tr_val,
        test_size=val_fraction_of_train,
        random_state=random_state,
        stratify=strat_mid,
    )
    return df_train, df_val, df_test
