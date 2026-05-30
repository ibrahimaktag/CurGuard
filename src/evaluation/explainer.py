"""SHAP-based explainability for specialist models.

Dashboard ve inference log için top-k feature importance.
"""

from typing import Any

import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    feature_names: list[str] | None = None,
) -> Any:
    """Create SHAP explainer for a Keras model.

    Uses GradientExplainer (or DeepExplainer) for neural networks.
    Background data should be a representative sample (e.g. 100-500 rows).

    Args:
        model: Compiled Keras model with predict.
        X_background: Background dataset (n_background, n_features).
        feature_names: Optional feature names for interpretation.

    Returns:
        SHAP explainer object.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. pip install shap")

    if len(X_background) > 200:
        X_background = X_background[np.random.choice(len(X_background), 200, replace=False)]

    try:
        explainer = shap.GradientExplainer(model, X_background)
    except Exception:
        try:
            explainer = shap.DeepExplainer(model, X_background)
        except Exception as e:
            logger.warning("SHAP GradientExplainer failed, trying KernelExplainer: %s", e)
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x, verbose=0),
                X_background[:50],
            )
    return explainer


def get_shap_values(
    explainer: Any,
    X: np.ndarray,
    n_samples: int | None = 100,
) -> np.ndarray:
    """Compute SHAP values for input samples.

    Args:
        explainer: SHAP explainer from create_shap_explainer.
        X: Input samples (n_samples, n_features).
        n_samples: Max samples to explain (for speed). None = all.

    Returns:
        SHAP values array (n_samples, n_features) or (n_samples, n_features, n_classes).
    """
    if n_samples is not None and len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]

    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        logger.warning("SHAP values computation failed: %s", e)
        return np.zeros((len(X), X.shape[1]))

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = np.mean(np.abs(shap_values), axis=0)
    return np.asarray(shap_values)


def get_top_k_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Get top-k features by mean absolute SHAP value.

    Args:
        shap_values: (n_samples, n_features) from get_shap_values.
        feature_names: Feature names.
        top_k: Number of top features. From config if None.

    Returns:
        List of (feature_name, mean_abs_shap) sorted descending.
    """
    if top_k is None:
        top_k = load_config("ensemble").get("xai", {}).get("top_k_features", 10)

    if len(feature_names) != shap_values.shape[1]:
        feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_k]
    return [(feature_names[i], float(mean_abs[i])) for i in top_indices]


def explain_prediction(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    X_background: np.ndarray | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """One-shot: create explainer, compute SHAP, return top features.

    Args:
        model: Keras model.
        X: Samples to explain (e.g. single sample or batch).
        feature_names: Feature names.
        X_background: Background for explainer. If None, uses X[:100].
        top_k: Number of top features to return.

    Returns:
        Dict with 'top_features': [(name, value), ...], 'shap_values': ndarray.
    """
    if X_background is None:
        X_background = X[: min(100, len(X))]
    if len(X_background) == 0:
        X_background = X

    explainer = create_shap_explainer(model, X_background, feature_names)
    shap_vals = get_shap_values(explainer, X, n_samples=min(50, len(X)))
    top = get_top_k_shap_features(shap_vals, feature_names, top_k=top_k)

    return {"top_features": top, "shap_values": shap_vals}
