"""Load trained specialist models and preprocessors for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib

from src.utils.config import get_project_root

Domain = Literal["iot", "cloud"]


def specialist_paths(domain: Domain) -> tuple[Path, Path]:
    """Return (model.keras, preprocessor.joblib) paths for a domain."""
    root = get_project_root()
    if domain == "iot":
        base = root / "artifacts" / "models" / "iot_specialist"
    else:
        base = root / "artifacts" / "models" / "cloud_specialist"
    return base / "model.keras", base / "preprocessor.joblib"


def load_specialist_bundle(domain: Domain) -> tuple[Any | None, Any | None]:
    """Load Keras model and fitted preprocessor if artifacts exist.

    Returns:
        (model, preprocessor) or (None, None) if model file is missing.
    """
    import tensorflow as tf

    model_path, prep_path = specialist_paths(domain)
    if not model_path.exists():
        return None, None
    model = tf.keras.models.load_model(model_path)
    prep = joblib.load(prep_path) if prep_path.exists() else None
    return model, prep
