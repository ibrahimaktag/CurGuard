"""Tests for router feature extraction (config mocked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import router as router_mod


@pytest.fixture
def mock_router_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal feature_mapping for router."""

    def _cfg() -> dict:
        return {
            "feature_mapping": {
                "dur": {"iot": "flow_duration", "cloud": "Flow Duration"},
                "flag": {"iot": "syn_flag_number", "cloud": "SYN Flag Count"},
            },
            "uncertainty_threshold": 0.75,
        }

    monkeypatch.setattr(router_mod, "_load_router_config", _cfg)


def test_extract_router_features_iot(mock_router_config: object) -> None:
    """IoT domain uses iot aliases."""
    df = pd.DataFrame({"flow_duration": [1.0, 2.0], "syn_flag_number": [0, 1]})
    out = router_mod.extract_router_features(df, domain="iot")
    assert list(out.columns) == ["dur", "flag"]
    np.testing.assert_array_almost_equal(out["dur"].values, [1.0, 2.0])


def test_extract_router_features_auto_prefers_first_match(mock_router_config: object) -> None:
    """Auto mode picks IoT names when both match (iot listed first in code)."""
    df = pd.DataFrame(
        {
            "flow_duration": [3.0],
            "Flow Duration": [99.0],
        }
    )
    out = router_mod.extract_router_features_auto(df)
    assert out["dur"].iloc[0] == 3.0
