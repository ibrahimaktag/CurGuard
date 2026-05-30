"""CLI: evaluate a trained IoT or Cloud specialist on the held-out test split."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import keras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_raw
from src.training.splits import split_train_val_test_df
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_specialist_artifact_dir(domain: str) -> Path:
    """Return artifact directory for the selected specialist domain."""
    return get_project_root() / "artifacts" / "models" / f"{domain}_specialist"


def _import_tensorflow_with_retry(max_attempts: int = 3, sleep_seconds: float = 1.5):
    """Import TensorFlow with a few retries for transient Windows DLL load errors."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return importlib.import_module("tensorflow")
        except Exception as exc:  # pragma: no cover - environment-dependent runtime failure
            last_error = exc
            logger.warning(
                "TensorFlow import failed (attempt %s/%s): %s",
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("TensorFlow import failed for an unknown reason.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate specialist on test split (same split logic as training scripts).",
    )
    p.add_argument(
        "--domain",
        choices=("iot", "cloud"),
        required=True,
        help="Which specialist to evaluate.",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Max rows per raw CSV (must match training intent for fair comparison).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: datasets.<domain>.random_state).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics JSON and predictions CSV (default: artifacts/evaluation).",
    )
    return p.parse_args()


def main() -> None:
    """Load data, test split, model, preprocessor; compute metrics and save artifacts."""
    args = parse_args()
    ds_cfg = load_config("datasets")[args.domain]
    rs = args.seed if args.seed is not None else ds_cfg.get("random_state", 42)
    target = ds_cfg["target_column"]
    test_r = float(ds_cfg.get("test_split", 0.2))
    val_r = float(ds_cfg.get("val_split", 0.1))

    logger.info("Loading raw data (domain=%s, nrows=%s)...", args.domain, args.nrows)
    df = load_dataset_raw(args.domain, nrows=args.nrows)

    df_train, _df_val, df_test = split_train_val_test_df(
        df,
        target_column=target,
        test_size=test_r,
        val_size=val_r,
        random_state=rs,
    )

    if len(df_test) == 0:
        raise RuntimeError("Test split is empty.")

    artifact_dir = _get_specialist_artifact_dir(args.domain)
    prep_path = artifact_dir / "preprocessor.joblib"
    if not prep_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {prep_path}. Re-run train_{args.domain}.py to save it."
        )
    prep = joblib.load(prep_path)

    tf = _import_tensorflow_with_retry()
    import pandas as pd

    from src.evaluation.metrics import (
        compute_classification_metrics,
        measure_inference_latency,
    )

    model_path = artifact_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras.models.load_model(model_path)

    X_test, y_test = prep.transform(df_test, include_target=True)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    names = prep.class_names
    metrics = compute_classification_metrics(
        y_test,
        y_pred,
        y_proba=probs,
        class_names=names,
        average="weighted",
    )
    latency = measure_inference_latency(model, X_test)

    out_dir = args.output_dir or (get_project_root() / "artifacts" / "evaluation")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.domain

    summary = {
        "domain": tag,
        "n_test": int(len(y_test)),
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision"],
        "recall_weighted": metrics["recall"],
        "f1_weighted": metrics["f1"],
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "per_class_f1": metrics["per_class_f1"],
        "latency_ms": latency,
        "classification_report": metrics["classification_report"],
    }
    json_path = out_dir / f"{tag}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Metrics written to %s", json_path)

    cm_path = out_dir / f"{tag}_confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({"confusion_matrix": metrics["confusion_matrix"]}, f, indent=2)

    pred_path = out_dir / f"{tag}_predictions.csv"
    pred_cols = {f"proba_{i}": probs[:, i] for i in range(probs.shape[1])}
    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            **pred_cols,
        }
    ).to_csv(pred_path, index=False)
    logger.info("Predictions for dashboard: %s", pred_path)

    print(metrics["classification_report"])
    print("Latency (ms):", latency)


if __name__ == "__main__":
    main()
