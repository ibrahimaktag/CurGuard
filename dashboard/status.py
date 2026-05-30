"""Shared artifact path checks for Streamlit pages."""

from __future__ import annotations

from src.utils.config import get_project_root


def artifact_status() -> dict[str, bool]:
    """Return whether key model artifacts exist on disk."""
    root = get_project_root()
    return {
        "router": (root / "artifacts" / "models" / "router" / "router.pkl").exists(),
        "iot_specialist": (root / "artifacts" / "models" / "iot_specialist" / "model.keras").exists(),
        "cloud_specialist": (root / "artifacts" / "models" / "cloud_specialist" / "model.keras").exists(),
        "anomaly_iot": (root / "artifacts" / "models" / "anomaly_detector_iot" / "vae.keras").exists(),
        "anomaly_cloud": (root / "artifacts" / "models" / "anomaly_detector_cloud" / "vae.keras").exists(),
        "anomaly_legacy": (root / "artifacts" / "models" / "anomaly_detector" / "vae.keras").exists(),
    }
