"""Evaluation: metrics and SHAP explainability."""

from src.evaluation.explainer import (
    create_shap_explainer,
    explain_prediction,
    get_shap_values,
    get_top_k_shap_features,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    get_top_k_features_by_importance,
    measure_inference_latency,
)

__all__ = [
    "compute_classification_metrics",
    "measure_inference_latency",
    "get_top_k_features_by_importance",
    "create_shap_explainer",
    "get_shap_values",
    "get_top_k_shap_features",
    "explain_prediction",
]
