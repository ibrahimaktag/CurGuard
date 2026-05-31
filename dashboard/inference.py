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
    """Load PyTorch model and fitted preprocessor if artifacts exist.

    Returns:
        (model, preprocessor) or (None, None) if model file is missing.
    """
    import torch
    from src.utils.config import get_project_root

    root = get_project_root()
    if domain == "iot":
        model_path = root / "artifacts" / "evaluation" / "best_iot_model.pt"
        prep_path = root / "artifacts" / "models" / "iot_specialist" / "preprocessor.joblib"
        input_dim = 39
        from src.models.iot_specialist import IoTSpecialist
        model_cls = IoTSpecialist
    else:
        model_path = root / "artifacts" / "evaluation" / "best_cloud_model.pt"
        prep_path = root / "artifacts" / "models" / "cloud_specialist" / "preprocessor.joblib"
        input_dim = 71
        from src.models.cloud_specialist import CloudSpecialist
        model_cls = CloudSpecialist

    if not model_path.exists():
        return None, None

    prep = joblib.load(prep_path) if prep_path.exists() else None
    model = model_cls(input_dim=input_dim)
    try:
        model.load_model(model_path)
        model.eval()
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    return model, prep
