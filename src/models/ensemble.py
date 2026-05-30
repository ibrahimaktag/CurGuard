"""Ensemble: combines Router, Specialists, and Anomaly Detector.

Confidence-weighted fusion. Anomaly override when score exceeds threshold.
Supports dual anomaly models (IoT vs Cloud feature spaces) plus legacy single model.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.router import load_router, route
from src.models.anomaly_detector import AnomalyDetector
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _infer_n_samples(
    X_anomaly: np.ndarray | None,
    X_iot: np.ndarray | None,
    X_cloud: np.ndarray | None,
    df_router: pd.DataFrame | None,
) -> int:
    """Infer batch size from the first non-empty input."""
    for arr in (X_iot, X_cloud, X_anomaly):
        if arr is not None:
            return len(arr)
    if df_router is not None:
        return len(df_router)
    raise ValueError("Provide at least one of X_iot, X_cloud, X_anomaly, or df_router.")


def _combine_dual_anomaly_scores(
    routing: list[dict[str, Any]],
    s_iot: np.ndarray,
    s_cloud: np.ndarray,
) -> np.ndarray:
    """Pick or fuse per-sample anomaly scores using routed domain.

    Args:
        routing: Per-sample router output.
        s_iot: Scores from IoT anomaly model (zeros if unused).
        s_cloud: Scores from Cloud anomaly model.

    Returns:
        Combined score per sample.
    """
    n = len(routing)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dom = routing[i]["domain"]
        if dom == "iot":
            out[i] = s_iot[i]
        elif dom == "cloud":
            out[i] = s_cloud[i]
        else:
            out[i] = max(s_iot[i], s_cloud[i])
    return out


class Ensemble:
    """Ensemble of Router + IoT/Cloud Specialists + Anomaly Detector(s).

    Flow: df_router -> Router -> domain -> Specialist(s) -> fuse -> Anomaly override.
    """

    def __init__(
        self,
        router_model: Any = None,
        router_feature_names: list[str] | None = None,
        iot_model: Any = None,
        cloud_model: Any = None,
        anomaly_detector: AnomalyDetector | None = None,
        anomaly_detector_iot: AnomalyDetector | None = None,
        anomaly_detector_cloud: AnomalyDetector | None = None,
        iot_class_names: list[str] | None = None,
        cloud_class_names: list[str] | None = None,
    ) -> None:
        """Initialize ensemble with optional pre-loaded components.

        Args:
            router_model: Trained router. If None, loads from disk when needed.
            router_feature_names: Feature names for router input.
            iot_model: Trained IoT specialist. None = load or skip.
            cloud_model: Trained Cloud specialist. None = load or skip.
            anomaly_detector: Legacy single anomaly model (same X for all).
            anomaly_detector_iot: Anomaly model trained on IoT preprocessor features.
            anomaly_detector_cloud: Anomaly model trained on Cloud preprocessor features.
            iot_class_names: Class names for IoT specialist output.
            cloud_class_names: Class names for Cloud specialist output.
        """
        self._router = router_model
        self._router_feature_names = router_feature_names
        self._iot_model = iot_model
        self._cloud_model = cloud_model
        self._anomaly_legacy = anomaly_detector
        self._anomaly_iot = anomaly_detector_iot
        self._anomaly_cloud = anomaly_detector_cloud
        self._iot_class_names = iot_class_names or []
        self._cloud_class_names = cloud_class_names or []
        self._config = load_config("ensemble")

    def load_components(self) -> None:
        """Load router, specialists, and anomaly detector(s) from artifacts."""
        try:
            self._router, self._router_feature_names = load_router()
        except FileNotFoundError:
            logger.warning("Router not found, routing will be skipped.")

        try:
            from src.models.iot_specialist import IoTSpecialist

            iot = IoTSpecialist()
            self._iot_model = iot.load()
        except (FileNotFoundError, Exception) as e:
            logger.warning("IoT specialist not found: %s", e)

        try:
            from src.models.cloud_specialist import CloudSpecialist

            cloud = CloudSpecialist()
            self._cloud_model = cloud.load()
        except (FileNotFoundError, Exception) as e:
            logger.warning("Cloud specialist not found: %s", e)

        root = get_project_root() / "artifacts" / "models"
        names = self._config.get("anomaly_models", {})
        iot_dir = names.get("iot", "anomaly_detector_iot")
        cloud_dir = names.get("cloud", "anomaly_detector_cloud")

        self._anomaly_iot = self._try_load_anomaly(root / iot_dir)
        self._anomaly_cloud = self._try_load_anomaly(root / cloud_dir)

        if self._anomaly_iot is None and self._anomaly_cloud is None:
            self._anomaly_legacy = self._try_load_anomaly(root / "anomaly_detector")
            if self._anomaly_legacy is not None:
                logger.info("Loaded legacy single anomaly_detector model.")
        else:
            self._anomaly_legacy = None

    @staticmethod
    def _try_load_anomaly(path: Path) -> AnomalyDetector | None:
        """Load AnomalyDetector from directory if present."""
        vae = path / "vae.keras"
        if not vae.exists():
            return None
        try:
            det = AnomalyDetector()
            det.load(path)
            return det
        except Exception as e:
            logger.warning("Failed to load anomaly model from %s: %s", path, e)
            return None

    def predict(
        self,
        X_anomaly: np.ndarray | None = None,
        *,
        df_router: pd.DataFrame | None = None,
        X_iot: np.ndarray | None = None,
        X_cloud: np.ndarray | None = None,
        domain_override: str | None = None,
    ) -> dict[str, Any]:
        """Run full ensemble inference.

        Args:
            X_anomaly: Legacy single-model features for anomaly (same dim as training).
            df_router: Raw flow DataFrame for the router (row-aligned with tensors).
            X_iot: IoT-preprocessed features for specialist and IoT anomaly model.
            X_cloud: Cloud-preprocessed features for specialist and Cloud anomaly model.
            domain_override: Force domain (iot|cloud|both). If None, use router.

        Returns:
            Dict with predictions, probabilities, anomaly_scores, anomaly_override,
            routing, alerts.
        """
        n_samples = _infer_n_samples(X_anomaly, X_iot, X_cloud, df_router)
        anomaly_cfg = self._config.get("anomaly_integration", {})
        alert_cfg = self._config.get("alert_thresholds", {})

        routing = [{"domain": domain_override or "unknown", "confidence": 0.0}] * n_samples
        if self._router is not None and domain_override is None:
            try:
                if df_router is None:
                    logger.warning(
                        "df_router is None; router needs raw DataFrame. Using 'both' for all rows."
                    )
                    routing = [{"domain": "both", "confidence": 0.0}] * n_samples
                elif len(df_router) != n_samples:
                    raise ValueError(
                        f"df_router rows ({len(df_router)}) must match batch size ({n_samples})"
                    )
                else:
                    routing = route(df_router, model=self._router)
            except Exception as e:
                logger.warning("Router failed: %s", e)

        anomaly_scores = self._compute_anomaly_scores(
            n_samples=n_samples,
            routing=routing,
            X_anomaly=X_anomaly,
            X_iot=X_iot,
            X_cloud=X_cloud,
        )

        override_threshold = anomaly_cfg.get("override_threshold", 0.8)
        anomaly_override = anomaly_scores >= override_threshold

        probs_iot = np.zeros((n_samples, 2))
        probs_cloud = np.zeros((n_samples, 2))

        if self._iot_model is not None and X_iot is not None and len(X_iot) == n_samples:
            try:
                probs_iot = self._iot_model.predict(X_iot, verbose=0)
            except Exception as e:
                logger.warning("IoT specialist failed: %s", e)

        if self._cloud_model is not None and X_cloud is not None and len(X_cloud) == n_samples:
            try:
                probs_cloud = self._cloud_model.predict(X_cloud, verbose=0)
            except Exception as e:
                logger.warning("Cloud specialist failed: %s", e)

        predictions = np.zeros(n_samples, dtype=np.int32)
        probabilities = np.zeros((n_samples, max(probs_iot.shape[1], probs_cloud.shape[1], 2)))

        for i in range(n_samples):
            dom = routing[i]["domain"]

            if anomaly_override[i] and anomaly_cfg.get("mode") == "override":
                predictions[i] = -1
                probabilities[i, 0] = 1.0
                continue

            if dom == "iot" and probs_iot.shape[1] > 1:
                pred = np.argmax(probs_iot[i])
                predictions[i] = pred
                probabilities[i, : probs_iot.shape[1]] = probs_iot[i]
            elif dom == "cloud" and probs_cloud.shape[1] > 1:
                pred = np.argmax(probs_cloud[i])
                predictions[i] = pred
                probabilities[i, : probs_cloud.shape[1]] = probs_cloud[i]
            elif dom == "both":
                if probs_iot.shape[1] == probs_cloud.shape[1] and probs_iot.shape[1] > 1:
                    w_iot = 0.5
                    w_cloud = 0.5
                    combined = w_iot * probs_iot[i] + w_cloud * probs_cloud[i]
                    pred = np.argmax(combined)
                    predictions[i] = pred
                    probabilities[i, : len(combined)] = combined
                elif probs_iot.shape[1] > 1:
                    predictions[i] = np.argmax(probs_iot[i])
                    probabilities[i, : probs_iot.shape[1]] = probs_iot[i]
                elif probs_cloud.shape[1] > 1:
                    predictions[i] = np.argmax(probs_cloud[i])
                    probabilities[i, : probs_cloud.shape[1]] = probs_cloud[i]
            else:
                if probs_iot.shape[1] > 1:
                    predictions[i] = np.argmax(probs_iot[i])
                    probabilities[i, : probs_iot.shape[1]] = probs_iot[i]
                elif probs_cloud.shape[1] > 1:
                    predictions[i] = np.argmax(probs_cloud[i])
                    probabilities[i, : probs_cloud.shape[1]] = probs_cloud[i]

        alerts = self._build_alerts(
            predictions=predictions,
            anomaly_scores=anomaly_scores,
            routing=routing,
            probs_iot=probs_iot,
            probs_cloud=probs_cloud,
            alert_cfg=alert_cfg,
        )

        return {
            "predictions": predictions,
            "probabilities": probabilities if probabilities.size > 0 else None,
            "anomaly_scores": anomaly_scores,
            "anomaly_override": anomaly_override,
            "routing": routing,
            "alerts": alerts,
        }

    def _compute_anomaly_scores(
        self,
        n_samples: int,
        routing: list[dict[str, Any]],
        X_anomaly: np.ndarray | None,
        X_iot: np.ndarray | None,
        X_cloud: np.ndarray | None,
    ) -> np.ndarray:
        """Compute per-sample hybrid anomaly scores."""
        out = np.zeros(n_samples, dtype=np.float32)
        use_dual = self._anomaly_iot is not None or self._anomaly_cloud is not None

        if use_dual:
            s_iot = np.zeros(n_samples, dtype=np.float32)
            s_cloud = np.zeros(n_samples, dtype=np.float32)
            try:
                if self._anomaly_iot is not None and X_iot is not None and len(X_iot) == n_samples:
                    s_iot = self._anomaly_iot.predict(X_iot)
            except Exception as e:
                logger.warning("IoT anomaly detector failed: %s", e)
            try:
                if self._anomaly_cloud is not None and X_cloud is not None and len(X_cloud) == n_samples:
                    s_cloud = self._anomaly_cloud.predict(X_cloud)
            except Exception as e:
                logger.warning("Cloud anomaly detector failed: %s", e)
            return _combine_dual_anomaly_scores(routing, s_iot, s_cloud)

        if self._anomaly_legacy is not None and X_anomaly is not None and len(X_anomaly) == n_samples:
            try:
                return self._anomaly_legacy.predict(X_anomaly)
            except Exception as e:
                logger.warning("Anomaly detector failed: %s", e)
        elif self._anomaly_legacy is not None and X_anomaly is None:
            logger.warning("Legacy anomaly model loaded but X_anomaly not provided; scores are zero.")

        return out

    def _build_alerts(
        self,
        predictions: np.ndarray,
        anomaly_scores: np.ndarray,
        routing: list,
        probs_iot: np.ndarray,
        probs_cloud: np.ndarray,
        alert_cfg: dict,
    ) -> list[dict]:
        """Build alert dicts per sample."""
        n = len(predictions)
        attack_thresh = alert_cfg.get("attack_confidence", 0.7)
        anomaly_thresh = alert_cfg.get("anomaly_score", 0.5)
        low_conf_thresh = alert_cfg.get("low_confidence_warn", 0.5)

        alerts = []
        for i in range(n):
            a = {"attack_detected": False, "anomaly_detected": False, "low_confidence": False}
            if predictions[i] > 0:
                max_prob = max(
                    probs_iot[i].max() if len(probs_iot.shape) > 1 else 0,
                    probs_cloud[i].max() if len(probs_cloud.shape) > 1 else 0,
                )
                a["attack_detected"] = max_prob >= attack_thresh
            if anomaly_scores[i] >= anomaly_thresh:
                a["anomaly_detected"] = True
            if routing[i]["confidence"] < low_conf_thresh:
                a["low_confidence"] = True
            alerts.append(a)
        return alerts
