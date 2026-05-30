"""Tests for train/val/test DataFrame splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.training.splits import split_train_val_test_df


def test_split_train_val_test_df_shapes() -> None:
    """Train + val + test fractions should match config-style ratios."""
    df = pd.DataFrame(
        {
            "label": ["a"] * 50 + ["b"] * 50,
            "x": range(100),
        }
    )
    tr, va, te = split_train_val_test_df(
        df,
        target_column="label",
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )
    assert len(tr) + len(va) + len(te) == 100
    assert len(te) == 20
    assert len(va) == 10
    assert len(tr) == 70


def test_split_train_val_test_df_stratify_disabled_single_class() -> None:
    """Single-class data should still split without stratify."""
    df = pd.DataFrame({"label": ["a"] * 30, "x": range(30)})
    tr, va, te = split_train_val_test_df(
        df,
        target_column="label",
        test_size=0.2,
        val_size=0.1,
        random_state=0,
    )
    assert len(tr) + len(va) + len(te) == 30


def test_split_invalid_ratios() -> None:
    """Invalid ratio combinations must raise."""
    df = pd.DataFrame({"label": [0, 1], "x": [0, 1]})
    with pytest.raises(ValueError, match="must be < 1"):
        split_train_val_test_df(
            df,
            target_column="label",
            test_size=0.6,
            val_size=0.5,
            random_state=0,
        )
