"""Traffic Router: meta-classifier for IoT vs Cloud domain routing.

Uses LightGBM (or RandomForest) trained on a common feature space derived
from both datasets via an explicit feature mapping (router.yaml).

Both CIC-IoT2023 and CIC-IDS2018 have semantically equivalent columns but
with different names. The mapping extracts these into a shared representation
so the router can distinguish IoT vs Cloud traffic regardless of column naming.
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.utils.seed import get_seed, set_all_seeds

logger = get_logger(__name__)

Domain = Literal["iot", "cloud"]


def _load_router_config() -> dict:
    """Load router configuration."""
    return load_config("router")


def extract_router_features(
    df: pd.DataFrame,
    domain: Domain,
) -> pd.DataFrame:
    """Extract common router features from a domain-specific DataFrame.

    Uses the feature_mapping in router.yaml to rename domain-specific columns
    to canonical router feature names. Missing columns are filled with 0.

    Args:
        df: Raw or lightly preprocessed DataFrame from one domain.
        domain: 'iot' or 'cloud' — selects which column names to look up.

    Returns:
        DataFrame with canonical router feature columns only.
    """
    cfg = _load_router_config()
    mapping = cfg.get("feature_mapping", {})

    extracted: dict[str, pd.Series] = {}
    for common_name, aliases in mapping.items():
        src_col = aliases.get(domain)
        if src_col and src_col in df.columns:
            extracted[common_name] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
        else:
            if src_col:
                logger.debug(
                    "Router: column '%s' ('%s' in %s domain) not found, filling with 0",
                    common_name, src_col, domain,
                )
            extracted[common_name] = pd.Series(np.zeros(len(df)), dtype=np.float32)

    return pd.DataFrame(extracted, index=df.index)


def extract_router_features_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Extract router features from an unknown-domain DataFrame.

    Tries column names from both IoT and Cloud mappings. The first match wins.
    Used at inference time when the domain is not yet known.

    Args:
        df: Raw DataFrame with unknown domain.

    Returns:
        DataFrame with canonical router feature columns.
    """
    cfg = _load_router_config()
    mapping = cfg.get("feature_mapping", {})

    extracted: dict[str, pd.Series] = {}
    for common_name, aliases in mapping.items():
        found = False
        for domain_name in ("iot", "cloud"):
            src_col = aliases.get(domain_name)
            if src_col and src_col in df.columns:
                extracted[common_name] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
                found = True
                break
        if not found:
            extracted[common_name] = pd.Series(np.zeros(len(df)), dtype=np.float32)

    return pd.DataFrame(extracted, index=df.index)


def train_router(
    df_iot: pd.DataFrame,
    df_cloud: pd.DataFrame,
    save_path: Path | None = None,
) -> object:
    """Train the router on raw IoT and Cloud DataFrames.

    Extracts a common feature set from each domain using feature_mapping,
    then trains a binary classifier (0=IoT, 1=Cloud).

    Args:
        df_iot: Raw DataFrame from CIC-IoT2023.
        df_cloud: Raw DataFrame from CIC-IDS2018.
        save_path: Where to save the trained model. If None, uses default.

    Returns:
        Trained classifier model.
    """
    set_all_seeds()
    cfg = _load_router_config()
    algorithm = cfg.get("algorithm", "lightgbm")

    X_iot = extract_router_features(df_iot, domain="iot").values.astype(np.float32)
    X_cloud = extract_router_features(df_cloud, domain="cloud").values.astype(np.float32)
    feature_names = list(cfg.get("feature_mapping", {}).keys())

    logger.info(
        "Router training: IoT samples=%s, Cloud samples=%s, features=%s",
        len(X_iot), len(X_cloud), len(feature_names),
    )

    X = np.vstack([X_iot, X_cloud])
    y = np.array([0] * len(X_iot) + [1] * len(X_cloud))
    rng = np.random.default_rng(get_seed())
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if algorithm == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**cfg.get("lightgbm_params", {}))
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**cfg.get("random_forest_params", {}))

    model.fit(X, y)
    logger.info("Router trained: algorithm=%s, total samples=%s", algorithm, len(y))

    if save_path is None:
        root = get_project_root()
        save_path = root / "artifacts" / "models" / "router" / "router.pkl"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, save_path)
    logger.info("Router saved to %s", save_path)
    return model


def load_router(model_path: Path | None = None) -> tuple[object, list[str]]:
    """Load trained router from disk.

    Args:
        model_path: Path to saved model. If None, uses default artifacts path.

    Returns:
        (model, feature_names).

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if model_path is None:
        root = get_project_root()
        model_path = root / "artifacts" / "models" / "router" / "router.pkl"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Router model not found: {model_path}")

    data = joblib.load(model_path)
    return data["model"], data["feature_names"]


def route(
    df: pd.DataFrame,
    model: object | None = None,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Route each sample to IoT, Cloud, or both.

    Args:
        df: Raw DataFrame with unknown domain (column names from either dataset).
        model: Trained router model. If None, loads from disk.
        feature_names: Ignored (kept for backward compat); canonical names from config are used.

    Returns:
        List of dicts: [{"domain": "iot"|"cloud"|"both", "confidence": float}, ...]
    """
    cfg = _load_router_config()
    threshold = cfg.get("uncertainty_threshold", 0.75)

    if model is None:
        model, _ = load_router()

    X = extract_router_features_auto(df).values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    proba = model.predict_proba(X)
    confidence = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)

    domain_map = {0: "iot", 1: "cloud"}
    results = []
    for c, p in zip(confidence, pred):
        if c >= threshold:
            results.append({"domain": domain_map[p], "confidence": float(c)})
        else:
            results.append({"domain": "both", "confidence": float(c)})
            logger.debug("Low router confidence %.3f → fallback to both specialists", c)

    return results
