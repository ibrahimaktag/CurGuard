"""Tests for router feature_mapping validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.router_mapping_validate import (
    read_csv_column_names,
    validate_mapping_vs_columns,
)


def test_validate_mapping_vs_columns_ok() -> None:
    """No missing entries when all mapped names exist."""
    mapping = {
        "f1": {"iot": "a", "cloud": "B"},
        "f2": {"iot": "c", "cloud": "D"},
    }
    r = validate_mapping_vs_columns(mapping, {"a", "c"}, {"B", "D"})
    assert r.ok
    assert not r.missing_iot
    assert not r.missing_cloud


def test_validate_mapping_vs_columns_reports_missing() -> None:
    """Missing IoT and Cloud columns are listed."""
    mapping = {"f1": {"iot": "missing_iot", "cloud": "ok_cloud"}}
    r = validate_mapping_vs_columns(mapping, {"wrong"}, {"ok_cloud"})
    assert not r.ok
    assert any("missing_iot" in m for m in r.missing_iot)
    assert not r.missing_cloud


def test_read_csv_column_names_strips(tmp_path: Path) -> None:
    """Headers are read without loading rows."""
    p = tmp_path / "t.csv"
    p.write_text(" A , B \n1,2\n", encoding="utf-8")
    cols = read_csv_column_names(p)
    assert cols == {"A", "B"}
