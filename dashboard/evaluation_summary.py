"""Load saved evaluation JSON files for the Monitoring page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.config import get_project_root


def evaluation_dir() -> Path:
    """Return the default directory for evaluation artifacts."""
    return get_project_root() / "artifacts" / "evaluation"


def load_json_if_exists(name: str) -> dict[str, Any] | None:
    """Load a JSON file from the evaluation directory, or None if missing.

    Args:
        name: Filename (e.g. ``iot_metrics.json``).

    Returns:
        Parsed dict or None.
    """
    path = evaluation_dir() / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_evaluation_summaries() -> dict[str, dict[str, Any] | None]:
    """Load known evaluation outputs if present.

    Returns:
        Keys: ``iot``, ``cloud``, ``router``; values are dicts or None.
    """
    return {
        "iot": load_json_if_exists("iot_metrics.json"),
        "cloud": load_json_if_exists("cloud_metrics.json"),
        "router": load_json_if_exists("router_metrics.json"),
    }
