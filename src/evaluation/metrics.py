"""Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, per-class.

Tüm metrikler bu modülden gelmeli (cursorrules).
"""

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    average: str = "weighted",
) -> dict[str, Any]:
    """Compute full classification metrics suite.

    Args:
        y_true: Ground truth labels (integer).
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (n_samples, n_classes). Required for ROC/PR-AUC.
        class_names: Optional class names for per-class report.
        average: 'micro', 'macro', or 'weighted' for F1/Precision/Recall.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc,
        per_class_f1, confusion_matrix, classification_report.
    """
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None and y_proba.shape[1] == n_classes:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
            )
        except ValueError:
            metrics["roc_auc"] = 0.0
            logger.warning("ROC-AUC computation failed (e.g. single class in subset)")

        try:
            from sklearn.preprocessing import label_binarize

            y_bin = label_binarize(y_true, classes=np.unique(y_true))
            if y_bin.shape[1] == y_proba.shape[1]:
                metrics["pr_auc"] = float(
                    average_precision_score(y_bin, y_proba, average=average)
                )
            else:
                metrics["pr_auc"] = 0.0
        except (ValueError, ImportError):
            metrics["pr_auc"] = 0.0
            logger.warning("PR-AUC computation failed")
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    sorted_classes = np.unique(np.concatenate([y_true, y_pred]))
    metrics["per_class_f1"] = {}
    for i, cls in enumerate(sorted_classes):
        name = class_names[int(cls)] if class_names and int(cls) < len(class_names) else str(cls)
        metrics["per_class_f1"][name] = float(per_class_f1[i])

    # Pass explicit labels so classification_report never raises when a class
    # is missing from predictions (e.g. degenerate test splits or dry-run data).
    report_labels = list(range(len(class_names))) if class_names else None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        labels=report_labels,
    )

    return metrics


def measure_inference_latency(
    model: Any,
    X: np.ndarray,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure model inference latency in milliseconds.

    Args:
        model: Keras model with predict() method.
        X: Sample batch (same shape as production input).
        n_warmup: Warmup iterations (discarded).
        n_runs: Number of timed runs.

    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms, p95_ms.
    """
    if len(X) == 0:
        return {"mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}

    for _ in range(n_warmup):
        model.predict(X[: min(32, len(X))], verbose=0)

    latencies_ms: list[float] = []
    batch = X[: min(256, len(X))]
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(batch, verbose=0)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _compute_latency_stats(latencies_ms: list) -> dict[str, float]:
    """Compute descriptive statistics over a list of latency measurements.

    Args:
        latencies_ms: Per-run latency values in milliseconds.

    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms, p95_ms.
    """
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms":  float(np.std(arr)),
        "min_ms":  float(np.min(arr)),
        "max_ms":  float(np.max(arr)),
        "p95_ms":  float(np.percentile(arr, 95)),
    }


def measure_inference_latency_torch(
    model: nn.Module,
    input_dim: int,
    batch_sizes: tuple[int, ...] = (1, 32),
    n_warmup: int = 20,
    n_runs: int = 200,
    device: Optional[torch.device] = None,
) -> dict[int, dict[str, float]]:
    """Measure per-batch-size inference latency for a PyTorch model.

    Uses torch.cuda.Event timing for GPU measurements, which correctly
    accounts for CUDA's asynchronous execution model.  A naive
    time.perf_counter() call only captures the CPU time to *enqueue*
    the kernel — it systematically under-reports real GPU latency.

    On CPU the function falls back to time.perf_counter() (valid because
    CPU execution is synchronous).

    The model is switched to eval() mode for the duration of the
    measurement and restored to its original training state afterward.
    Gradients are disabled via torch.no_grad() to avoid phantom VRAM
    allocation and give realistic production-inference timings.

    Args:
        model:       Trained PyTorch nn.Module (CloudSpecialist / IoTSpecialist).
        input_dim:   Number of input features — used to construct random
                     dummy tensors of the correct shape.
        batch_sizes: Tuple of batch sizes to benchmark independently.
                     Use (1,) for SOC real-time single-packet simulation;
                     use (32,) for batch-processing throughput; pass both
                     to obtain a complete latency profile.
        n_warmup:    Number of discarded warm-up passes.  GPU JIT
                     compilation happens here, so timings are not polluted.
        n_runs:      Number of timed repetitions per batch size.  Higher
                     values reduce measurement variance (200 recommended).
        device:      Target device.  If None, uses model's current device
                     (detected via next(model.parameters()).device).

    Returns:
        Nested dict keyed by batch size::

            {
                1:  {"mean_ms": 0.42, "std_ms": 0.03, ...},
                32: {"mean_ms": 1.87, "std_ms": 0.11, ...},
            }

        Each inner dict contains: mean_ms, std_ms, min_ms, max_ms, p95_ms.
    """
    # --- Resolve device ---
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    # --- Preserve model training state ---
    was_training = model.training
    model.eval()

    results: dict[int, dict[str, float]] = {}

    try:
        with torch.no_grad():
            for batch_size in batch_sizes:
                dummy = torch.randn(batch_size, input_dim, device=device)
                latencies_ms: list[float] = []

                # --- Warm-up (discard JIT / cuDNN auto-tune overhead) ---
                for _ in range(n_warmup):
                    _ = model(dummy)
                if use_cuda:
                    torch.cuda.synchronize(device)

                # --- Timed runs ---
                if use_cuda:
                    # CUDA event timing: captures true GPU wall-clock time
                    # including kernel execution, not just kernel submission.
                    starter = torch.cuda.Event(enable_timing=True)
                    ender   = torch.cuda.Event(enable_timing=True)

                    for _ in range(n_runs):
                        starter.record()
                        _ = model(dummy)
                        ender.record()
                        # Block CPU until GPU finishes this run before reading
                        # the elapsed time — essential for correct measurement.
                        torch.cuda.synchronize(device)
                        latencies_ms.append(starter.elapsed_time(ender))
                else:
                    # CPU path: operations are synchronous, perf_counter is valid.
                    for _ in range(n_runs):
                        t0 = time.perf_counter()
                        _ = model(dummy)
                        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

                results[batch_size] = _compute_latency_stats(latencies_ms)
                logger.info(
                    "Latency [batch=%d, device=%s]: mean=%.3f ms  p95=%.3f ms",
                    batch_size, device, results[batch_size]["mean_ms"],
                    results[batch_size]["p95_ms"],
                )
    finally:
        # Always restore original mode even if an exception occurs
        if was_training:
            model.train()

    return results


def measure_inference_latency_auto(
    model: Any,
    input_dim_or_X: "int | np.ndarray",
    batch_sizes: tuple[int, ...] = (1, 32),
    n_warmup: int = 20,
    n_runs: int = 200,
    device: Optional[torch.device] = None,
) -> dict[int, dict[str, float]]:
    """Framework-agnostic latency dispatcher.

    Routes to the correct timing implementation based on model type:

    * ``torch.nn.Module``  ->  :func:`measure_inference_latency_torch`
      (CUDA event timing, CPU fallback)
    * Everything else      ->  :func:`measure_inference_latency`
      (Keras/TF ``.predict()`` with ``time.perf_counter``)

    This keeps the Ensemble evaluation loop simple: call this one function
    for CloudSpecialist, IoTSpecialist, and AnomalyDetector (VAE) alike.

    Args:
        model:             PyTorch nn.Module **or** Keras model.
        input_dim_or_X:    • int  — feature dimensionality (PyTorch path).
                           • np.ndarray — pre-built input array (Keras path).
        batch_sizes:       Batch sizes to benchmark.  Applied only on the
                           PyTorch path; the Keras path uses the shape of X.
        n_warmup:          Warm-up iterations (discarded).
        n_runs:            Timed repetitions.
        device:            Target device (PyTorch only; ignored for Keras).

    Returns:
        Nested dict ``{batch_size: {mean_ms, std_ms, min_ms, max_ms, p95_ms}}``.
        For the Keras path the dict has a single key equal to ``len(X)``.
    """
    if isinstance(model, nn.Module):
        # PyTorch specialist model
        if not isinstance(input_dim_or_X, int):
            raise TypeError(
                "For PyTorch models, input_dim_or_X must be an int (feature dimension)."
            )
        return measure_inference_latency_torch(
            model=model,
            input_dim=input_dim_or_X,
            batch_sizes=batch_sizes,
            n_warmup=n_warmup,
            n_runs=n_runs,
            device=device,
        )
    else:
        # Keras / TF model (AnomalyDetector VAE)
        if isinstance(input_dim_or_X, int):
            raise TypeError(
                "For Keras models, input_dim_or_X must be a numpy array (X)."
            )
        X = input_dim_or_X
        keras_stats = measure_inference_latency(
            model=model, X=X, n_warmup=n_warmup, n_runs=n_runs
        )
        # Wrap in the same nested-dict structure for a uniform return type
        return {len(X): keras_stats}



def get_top_k_features_by_importance(
    importance_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k features by absolute importance (for SHAP/other explainers).

    Args:
        importance_values: Per-feature importance (e.g. mean |SHAP|).
        feature_names: Names of features.

    Returns:
        List of (feature_name, importance) sorted descending.
    """
    if len(feature_names) != len(importance_values):
        feature_names = [f"f{i}" for i in range(len(importance_values))]
    abs_imp = np.abs(importance_values)
    top_indices = np.argsort(abs_imp)[::-1][:top_k]
    return [(feature_names[i], float(importance_values[i])) for i in top_indices]
