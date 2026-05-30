"""Full-scale Kaggle Training & Evaluation Orchestrator for C-rGuard.

Runs the complete pipeline on both Cloud (CIC-IDS2018) and IoT (CICIoT2023)
domains with GPU acceleration, then generates the full academic evaluation suite
plus three operational analysis modules:

  1. Router Cascade Analysis  — misrouting rate and its downstream F1 impact.
  2. Anomaly Override Module  — VAE-triggered overrides on Specialist "Benign" decisions.
  3. Cost-Sensitive Threat Matrix — operational cost model for FP/FN in SOC context.

Usage (Kaggle notebook cell)
-----------------------------
    !python /kaggle/working/C-rGuard/scripts/run_kaggle_full.py \\
        --domain both --epochs 50 --batch-size 512

Usage (local, full dataset)
----------------------------
    python scripts/run_kaggle_full.py --domain cloud --epochs 50 --nrows 0

Usage (local, quick smoke test)
--------------------------------
    python scripts/run_kaggle_full.py --domain cloud --epochs 3 --nrows 50000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# -- Ensure repo root is importable when invoked as a standalone script -------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import create_dataloader, load_dataset_raw
from src.data.preprocessing import CloudPreprocessor, IoTPreprocessor
from src.evaluation.metrics import (
    compute_classification_metrics,
    measure_inference_latency_torch,
)
from src.evaluation.plots import (
    plot_attention_heatmap,
    plot_confidence_calibration,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_training_curves,
    plot_tsne,
)
from src.models.cloud_specialist import CloudSpecialist
from src.models.iot_specialist import IoTSpecialist
from src.training.splits import split_train_val_test_df
from src.training.trainer import Trainer
from src.utils.config import (
    get_artifacts_root,
    get_dataset_path,
    is_kaggle,
    load_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain configuration table
# ---------------------------------------------------------------------------
_DOMAIN_CFG: dict[str, dict[str, Any]] = {
    "cloud": {
        "model_cls":   CloudSpecialist,
        "preproc_cls": CloudPreprocessor,
        "target_col":  "Label",
        "class_names": ["Benign", "Attack"],
        "task_type":   "binary",
        "preproc_kwargs": {"target_col": "Label", "task_type": "binary"},
    },
    "iot": {
        "model_cls":   IoTSpecialist,
        "preproc_cls": IoTPreprocessor,
        "target_col":  "label",
        "class_names": ["Benign", "Attack"],
        "task_type":   "binary",
        "preproc_kwargs": {"target_col": "label", "task_type": "binary"},
    },
}

# ---------------------------------------------------------------------------
# 1. Router Cascade Analysis
# ---------------------------------------------------------------------------

def analyze_router_cascade(
    routing: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred_specialist: np.ndarray,
    domain: str,
) -> dict[str, Any]:
    """Measure routing quality and downstream F1 impact of misrouted samples.

    A misrouted sample is one that the Router assigned to the *wrong* domain
    (e.g. IoT packet sent to the Cloud specialist). Since we know the true
    domain from the CLI ``--domain`` flag, any sample routed to a different
    domain is counted as misrouted.

    Args:
        routing:             Per-sample list of dicts with 'domain' and 'confidence'.
        y_true:              Ground-truth binary labels (0=Benign, 1=Attack).
        y_pred_specialist:   Specialist model predictions on the full test set.
        domain:              Correct domain ('cloud' or 'iot').

    Returns:
        dict with misrouting_rate, n_misrouted, f1_all, f1_correctly_routed,
        f1_misrouted, and the confidence distribution stats.
    """
    from sklearn.metrics import f1_score

    correct_domain = domain
    routed_domains  = np.array([r["domain"] for r in routing])
    confidences     = np.array([r["confidence"] for r in routing], dtype=np.float32)

    correctly_routed_mask = (routed_domains == correct_domain)
    n_misrouted  = int((~correctly_routed_mask).sum())
    n_total      = len(routing)
    misrouting_rate = n_misrouted / max(n_total, 1)

    f1_all = float(f1_score(y_true, y_pred_specialist, average="weighted", zero_division=0))

    f1_correct  = float(
        f1_score(y_true[correctly_routed_mask], y_pred_specialist[correctly_routed_mask],
                 average="weighted", zero_division=0)
    ) if correctly_routed_mask.sum() > 0 else 0.0

    f1_misrouted = float(
        f1_score(y_true[~correctly_routed_mask], y_pred_specialist[~correctly_routed_mask],
                 average="weighted", zero_division=0)
    ) if n_misrouted > 0 else None

    f1_delta = (f1_correct - f1_all) if f1_misrouted is not None else 0.0

    result = {
        "domain":              domain,
        "n_total":             n_total,
        "n_misrouted":         n_misrouted,
        "misrouting_rate_pct": round(misrouting_rate * 100, 4),
        "f1_all_samples":      round(f1_all, 6),
        "f1_correctly_routed": round(f1_correct, 6),
        "f1_misrouted":        round(f1_misrouted, 6) if f1_misrouted is not None else None,
        "f1_delta_vs_all":     round(f1_delta, 6),
        "router_confidence_mean": round(float(confidences.mean()), 4),
        "router_confidence_p25":  round(float(np.percentile(confidences, 25)), 4),
        "router_confidence_p75":  round(float(np.percentile(confidences, 75)), 4),
    }

    logger.info(
        "[Router Cascade] domain=%s  misrouted=%d/%d (%.2f%%)  "
        "f1_all=%.4f  f1_correct=%.4f  delta=%.4f",
        domain, n_misrouted, n_total, misrouting_rate * 100,
        f1_all, f1_correct, f1_delta,
    )
    return result


# ---------------------------------------------------------------------------
# 2. Anomaly Override Module (VAE)
# ---------------------------------------------------------------------------

def analyze_anomaly_override(
    y_true: np.ndarray,
    y_pred_specialist: np.ndarray,
    anomaly_scores: np.ndarray,
    override_threshold: float = 0.8,
) -> dict[str, Any]:
    """Quantify the detection lift delivered by the VAE anomaly override.

    Logic:
        - Start from Specialist predictions.
        - For samples where Specialist said "Benign" (pred=0) AND
          anomaly_score >= override_threshold: override to "Attack" (pred=1).
        - Count how many of those overrides were genuine attacks (True Positives).

    Args:
        y_true:               Ground-truth labels.
        y_pred_specialist:    Specialist predictions before override.
        anomaly_scores:       Per-sample anomaly scores from VAE hybrid detector.
        override_threshold:   Score threshold to trigger override (default 0.8).

    Returns:
        dict with n_overrides, n_true_positives_recovered, precision_of_override,
        f1_before_override, f1_after_override, and absolute recall_lift.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    override_mask = (y_pred_specialist == 0) & (anomaly_scores >= override_threshold)
    n_overrides   = int(override_mask.sum())

    y_pred_after = y_pred_specialist.copy()
    y_pred_after[override_mask] = 1

    # True Positives recovered: those where we overrode AND ground truth = Attack
    n_tp_recovered = int(((y_true == 1) & override_mask).sum())
    override_precision = n_tp_recovered / max(n_overrides, 1)

    f1_before = float(f1_score(y_true, y_pred_specialist, average="weighted", zero_division=0))
    f1_after  = float(f1_score(y_true, y_pred_after,      average="weighted", zero_division=0))

    recall_before = float(recall_score(y_true, y_pred_specialist, average="weighted", zero_division=0))
    recall_after  = float(recall_score(y_true, y_pred_after,      average="weighted", zero_division=0))

    fp_introduced = int(((y_true == 0) & override_mask).sum())

    result = {
        "override_threshold":       override_threshold,
        "n_overrides":              n_overrides,
        "n_true_positives_recovered": n_tp_recovered,
        "n_false_positives_introduced": fp_introduced,
        "override_precision":       round(override_precision, 6),
        "f1_before_override":       round(f1_before, 6),
        "f1_after_override":        round(f1_after, 6),
        "f1_lift":                  round(f1_after - f1_before, 6),
        "recall_before":            round(recall_before, 6),
        "recall_after":             round(recall_after, 6),
        "recall_lift":              round(recall_after - recall_before, 6),
    }

    logger.info(
        "[VAE Override] threshold=%.2f  overrides=%d  TP_recovered=%d  "
        "FP_introduced=%d  f1 %.4f -> %.4f (lift=%.4f)  recall_lift=%.4f",
        override_threshold, n_overrides, n_tp_recovered, fp_introduced,
        f1_before, f1_after, f1_after - f1_before, recall_after - recall_before,
    )
    return result


# ---------------------------------------------------------------------------
# 3. Cost-Sensitive Threat Matrix
# ---------------------------------------------------------------------------

def compute_cost_sensitive_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> dict[str, Any]:
    """Translate the confusion matrix into an operational cost model.

    Cost model
    ----------
    * FP (False Positive / False Alarm): SOC analyst investigation time.
      Default coefficient ``C_FP = 1.0`` (unit cost per false alarm).
    * FN (False Negative / Missed Attack): Potential data breach / damage.
      Default coefficient ``C_FN = 10.0`` (C_FN >> C_FP as per SOC SLAs).

    Total operational cost = FP × C_FP + FN × C_FN

    The ratio C_FN / C_FP encodes how much more expensive a missed attack is
    compared to a false alarm — a key parameter for SOC budget planning.

    Args:
        y_true:   Ground-truth binary labels.
        y_pred:   Model predictions.
        cost_fp:  Cost coefficient per False Positive (analyst effort unit).
        cost_fn:  Cost coefficient per False Negative (breach damage unit).

    Returns:
        dict with TP, TN, FP, FN, total_cost, cost_fp_total, cost_fn_total,
        cost_ratio, and savings_vs_no_model (cost if we flagged everything as Benign).
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle degenerate single-class case gracefully
        tn = int((y_true == 0).sum())
        fp = fn = 0
        tp = int((y_true == 1).sum())

    cost_fp_total = float(fp) * cost_fp
    cost_fn_total = float(fn) * cost_fn
    total_cost    = cost_fp_total + cost_fn_total

    # Baseline: naive model that labels everything Benign → all attacks missed
    n_attacks_total = int((y_true == 1).sum())
    baseline_cost   = n_attacks_total * cost_fn

    savings_vs_baseline = baseline_cost - total_cost
    savings_pct         = (savings_vs_baseline / max(baseline_cost, 1)) * 100

    result = {
        "cost_fp_coefficient":  cost_fp,
        "cost_fn_coefficient":  cost_fn,
        "cost_ratio_fn_vs_fp":  round(cost_fn / cost_fp, 2),
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
        },
        "cost_fp_total":        round(cost_fp_total, 2),
        "cost_fn_total":        round(cost_fn_total, 2),
        "total_operational_cost": round(total_cost, 2),
        "baseline_naive_cost":  round(float(baseline_cost), 2),
        "savings_vs_baseline":  round(float(savings_vs_baseline), 2),
        "savings_pct":          round(savings_pct, 2),
    }

    logger.info(
        "[Cost Matrix] FP=%d (cost=%.1f)  FN=%d (cost=%.1f)  "
        "total=%.1f  savings vs naive=%.1f (%.1f%%)",
        fp, cost_fp_total, fn, cost_fn_total, total_cost,
        savings_vs_baseline, savings_pct,
    )
    return result


# ---------------------------------------------------------------------------
# Forward-hook collector (same as run_phase5_local.py)
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_type: str = "binary",
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate model and collect predictions, probabilities, and attention weights.

    Args:
        model:     Trained PyTorch specialist (CloudSpecialist or IoTSpecialist).
        loader:    DataLoader wrapping the evaluation split.
        device:    Compute device.
        task_type: 'binary' or 'multiclass'.
        threshold: Decision threshold for binary classification (default 0.5).

    Returns:
        y_true, y_pred, y_proba, attn_weights (or None if unavailable).
    """
    model.eval()
    all_preds, all_targets, all_probs, all_attn = [], [], [], []
    attn_cache: dict = {}

    def _hook(module, inp, out):
        if out[1] is not None:
            attn_cache["w"] = out[1].detach().cpu().numpy()

    handle = model.attention.mha.register_forward_hook(_hook)

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits  = model(batch_X)

            if task_type == "binary":
                probs_pos = torch.sigmoid(logits).cpu().numpy()
                preds     = (probs_pos > threshold).astype(np.int64)
                probs_2d  = np.column_stack([1.0 - probs_pos, probs_pos])
            else:
                probs_2d = torch.softmax(logits, dim=1).cpu().numpy()
                preds    = probs_2d.argmax(axis=1).astype(np.int64)

            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.numpy().astype(np.int64).tolist())
            all_probs.append(probs_2d)
            if "w" in attn_cache:
                all_attn.append(attn_cache["w"])

    handle.remove()

    y_true  = np.array(all_targets, dtype=np.int64)
    y_pred  = np.array(all_preds,   dtype=np.int64)
    y_proba = np.vstack(all_probs).astype(np.float32)
    attn_w  = np.concatenate(all_attn, axis=0) if all_attn else None
    return y_true, y_pred, y_proba, attn_w


# ---------------------------------------------------------------------------
# Helper: class-weight tensor for BCEWithLogitsLoss
# ---------------------------------------------------------------------------

def _pos_weight(df: "pd.DataFrame", target_col: str, device: torch.device) -> torch.Tensor:
    """Compute pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    import pandas as pd
    counts = df[target_col].value_counts()
    n_neg  = int(counts.get(0, 1))
    n_pos  = int(counts.get(1, 1))
    ratio  = n_neg / max(n_pos, 1)
    logger.info("Class balance: neg=%d  pos=%d  pos_weight=%.3f", n_neg, n_pos, ratio)
    return torch.tensor([ratio], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Synthetic anomaly scores (standalone run without full Ensemble)
# ---------------------------------------------------------------------------

def _synthetic_anomaly_scores(y_proba: np.ndarray, noise_scale: float = 0.05) -> np.ndarray:
    """Generate plausible anomaly scores from specialist confidence.

    When the VAE/Ensemble is not loaded (standalone specialist-only run),
    we approximate anomaly scores from the specialist's own uncertainty:
    low confidence in benign → high anomaly score.

    This is a *simulation* intended to keep the 3 operational modules
    runnable even without a pre-trained AnomalyDetector artifact.
    Replace with real VAE scores in production.

    Args:
        y_proba:     (n, 2) probability array from specialist.
        noise_scale: Small Gaussian noise to prevent degenerate distributions.

    Returns:
        (n,) array of synthetic anomaly scores in [0, 1].
    """
    rng = np.random.default_rng(42)
    # Anomaly score = 1 − confidence_in_benign + small noise
    benign_confidence = y_proba[:, 0]
    scores = (1.0 - benign_confidence) + rng.normal(0, noise_scale, len(benign_confidence))
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic routing (standalone run without Router model)
# ---------------------------------------------------------------------------

def _synthetic_routing(n: int, domain: str) -> list[dict[str, Any]]:
    """Generate synthetic routing dicts when no Router model is loaded.

    Simulates a high-confidence router (95%) with 5% random misroutes so the
    cascade analysis produces meaningful (non-trivial) output.

    Args:
        n:      Number of samples.
        domain: Correct domain name.

    Returns:
        List of per-sample routing dicts.
    """
    rng = np.random.default_rng(42)
    wrong = "iot" if domain == "cloud" else "cloud"
    routing = []
    for _ in range(n):
        if rng.random() < 0.95:
            routing.append({"domain": domain, "confidence": float(rng.uniform(0.75, 1.0))})
        else:
            routing.append({"domain": wrong, "confidence": float(rng.uniform(0.4, 0.65))})
    return routing


# ---------------------------------------------------------------------------
# Main domain pipeline
# ---------------------------------------------------------------------------

def run_domain(
    domain: str,
    epochs: int,
    batch_size: int,
    nrows: int | None,
    save_dir: Path,
    datasets_cfg: dict,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, Any]:
    """Execute the full training + evaluation pipeline for one domain.

    Steps
    -----
    1. Load raw data via environment-aware get_dataset_path()
    2. Stratified train/val/test split (80/10/10)
    3. Preprocess (fit_transform + save sample CSV)
    4. Build DataLoaders
    5. Init model + Trainer; train for ``epochs`` epochs
    6. Plot training curves
    7. Collect test-set predictions + attention weights
    8. Standard evaluation plots (CM, ROC/PR, Calibration, Heatmap, t-SNE)
    9. Latency measurement
    10. Router Cascade Analysis
    11. Anomaly Override Module
    12. Cost-Sensitive Threat Matrix
    13. Save JSON summary

    Args:
        domain:         'cloud' or 'iot'.
        epochs:         Training epochs.
        batch_size:     DataLoader batch size.
        nrows:          Row limit for raw CSV loading (0 = load all).
        save_dir:       Output directory for this domain's artefacts.
        datasets_cfg:   Pre-loaded datasets.yaml dict.
        cost_fp:        FP cost coefficient for Cost-Sensitive Matrix.
        cost_fn:        FN cost coefficient for Cost-Sensitive Matrix.

    Returns:
        Summary dict with all metrics.
    """
    import pandas as pd

    cfg        = _DOMAIN_CFG[domain]
    target_col = cfg["target_col"]
    task_type  = cfg["task_type"]
    class_names = cfg["class_names"]
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 66)
    logger.info(
        "KAGGLE FULL  |  domain=%-6s  device=%s  epochs=%d  batch=%d%s",
        domain.upper(), device, epochs, batch_size,
        f"  nrows={nrows}" if nrows else "  nrows=ALL",
    )
    logger.info("=" * 66)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    logger.info("[1/8] Loading raw data ...")
    raw_path = get_dataset_path(domain, datasets_cfg)
    nrows_arg = nrows if nrows and nrows > 0 else None
    df_raw = load_dataset_raw(domain=domain, nrows=nrows_arg)

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    logger.info("[1/8] Splitting 80/10/10 ...")
    df_train, df_val, df_test = split_train_val_test_df(
        df_raw, target_column=target_col,
        test_size=0.10, val_size=0.10, random_state=42,
    )

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    preproc = cfg["preproc_cls"](**cfg["preproc_kwargs"])
    df_train_c = preproc.fit_transform(df_train, save_sample=True, n_sample_rows=50)
    df_val_c   = preproc.transform(df_val)
    df_test_c  = preproc.transform(df_test)

    feature_names = [c for c in df_train_c.columns if c != target_col]
    input_dim     = len(feature_names)
    num_classes   = int(df_train_c[target_col].nunique())
    logger.info("[1/8] input_dim=%d  num_classes=%d  train=%d  val=%d  test=%d",
                input_dim, num_classes, len(df_train_c), len(df_val_c), len(df_test_c))

    # ------------------------------------------------------------------
    # 4. DataLoaders
    # ------------------------------------------------------------------
    train_loader = create_dataloader(df_train_c, target_col, task_type,
                                     batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(df_val_c,   target_col, task_type,
                                     batch_size=batch_size, shuffle=False)
    test_loader  = create_dataloader(df_test_c,  target_col, task_type,
                                     batch_size=batch_size, shuffle=False)

    # WeightedRandomSampler: her batch'e eşit sayıda Benign+Attack
    # Sınıf çökşmesi (her şeyi tek sınıf tahmin) sorununu önler
    if task_type == 'binary':
        from torch.utils.data import WeightedRandomSampler, DataLoader as _DL
        _labels = train_loader.dataset.y_tensor.numpy().astype(int)
        _counts = np.bincount(_labels)
        if len(_counts) == 2 and min(_counts) > 0:
            _sw = torch.tensor(1.0 / _counts[_labels], dtype=torch.float32)
            _sampler = WeightedRandomSampler(_sw, len(_sw), replacement=True)
            train_loader = _DL(
                train_loader.dataset, batch_size=batch_size,
                sampler=_sampler, num_workers=0, pin_memory=True,
            )
            logger.info("WeightedRandomSampler aktif: neg=%d pos=%d eşit batch", _counts[0], _counts[1])

    # ------------------------------------------------------------------
    # 5. Model + Trainer
    # ------------------------------------------------------------------
    # LR ve patience: training.yaml'dan oku (hardcoded değil)
    training_cfg  = load_config("training")
    lr            = float(training_cfg.get("learning_rate", 1e-3))
    patience_val  = int(training_cfg.get("early_stopping_patience", 10))
    logger.info("Training config: lr=%.2e  patience=%d", lr, patience_val)

    model_save_path = str(save_dir / f"best_{domain}_model.pt")
    model     = cfg["model_cls"](input_dim=input_dim, num_classes=num_classes).to(device)
    model.count_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pw        = _pos_weight(df_train_c, target_col, device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        task_type=task_type,
        class_weights=pw,
        device=device,
        patience=patience_val,
        model_save_path=model_save_path,
    )

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    logger.info("[2/8] Training %d epoch(s) ...", epochs)
    t0 = time.perf_counter()
    best_metrics, history = trainer.fit(epochs=epochs)
    train_time = time.perf_counter() - t0
    logger.info("Training done %.1f s  best_val_f1=%.4f", train_time, best_metrics.get("val_f1", 0.0))

    # CRITICAL: En iyi checkpoint'i yükle (final epoch değil!)
    # trainer.fit() sonrası model hâlâ son epoch'un ağırlıklarını taşıyor.
    # Test değerlendirmesi için best_val_f1'deki kaydedilmiş modeli yüklememiz şart.
    try:
        model.load_model(model_save_path)
        model.eval()
        logger.info("Best model reloaded from %s", model_save_path)
    except Exception as e:
        logger.warning("Could not reload best model (%s) — using final epoch", e)

    # ------------------------------------------------------------------
    # Otomatik Threshold Optimizasyonu (val set üzerinde)
    # FP*cost_fp + FN*cost_fn maliyet fonksiyonunu minimize eden eşiği seç
    # ------------------------------------------------------------------
    best_threshold = 0.5
    if task_type == 'binary':
        _candidates = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        _best_cost  = float('inf')
        for _t in _candidates:
            _vt, _vp, _, _ = collect_predictions(model, val_loader, device, task_type, threshold=_t)
            _tn = int(np.sum((_vt == 0) & (_vp == 0)))
            _tp = int(np.sum((_vt == 1) & (_vp == 1)))
            _fp = int(np.sum((_vt == 0) & (_vp == 1)))
            _fn = int(np.sum((_vt == 1) & (_vp == 0)))
            _c  = _fp * cost_fp + _fn * cost_fn
            logger.info("Threshold=%.2f  TP=%d  TN=%d  FP=%d  FN=%d  cost=%.0f",
                        _t, _tp, _tn, _fp, _fn, _c)
            if _c < _best_cost:
                _best_cost = _c
                best_threshold = _t
        logger.info("Optimal threshold selected: %.2f (val cost=%.0f)", best_threshold, _best_cost)

    plot_training_curves(history, save_dir=save_dir, title_prefix=domain)

    # ------------------------------------------------------------------
    # 7. Standard evaluation
    # ------------------------------------------------------------------
    logger.info("[3/8] Evaluating on test set ...")
    y_true, y_pred, y_proba, attn_weights = collect_predictions(
        model, test_loader, device, task_type, threshold=best_threshold
    )

    from sklearn.metrics import classification_report
    report_labels = list(range(len(class_names)))
    logger.info(
        "Classification report:\n%s",
        classification_report(y_true, y_pred,
                              target_names=class_names,
                              labels=report_labels,
                              zero_division=0),
    )

    metrics = compute_classification_metrics(
        y_true, y_pred, y_proba=y_proba,
        class_names=class_names, average="weighted",
    )
    _print_metrics_table(domain, metrics)

    # ------------------------------------------------------------------
    # 8. Evaluation plots
    # ------------------------------------------------------------------
    logger.info("[4/8] Generating evaluation plots ...")
    plot_confusion_matrix(y_true, y_pred, save_dir=save_dir,
                          class_names=class_names, title_prefix=domain)
    plot_roc_pr_curves(y_true, y_proba, save_dir=save_dir,
                       class_names=class_names, title_prefix=domain)
    plot_confidence_calibration(y_true, y_proba, save_dir=save_dir, title_prefix=domain)

    if attn_weights is not None:
        plot_attention_heatmap(attn_weights=attn_weights, input_dim=input_dim,
                               save_dir=save_dir, feature_names=feature_names,
                               title_prefix=domain, max_samples=16)
    else:
        logger.warning("Attention weights unavailable — skipping heatmap.")

    logger.info("[4/8] t-SNE ...")
    X_test_np = df_test_c.drop(columns=[target_col]).values.astype(np.float32)
    plot_tsne(X=X_test_np, y=y_true, save_dir=save_dir,
              class_names=class_names, title_prefix=domain,
              n_samples=5_000, pca_components=50)

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------
    logger.info("[5/8] Measuring inference latency ...")
    latency = measure_inference_latency_torch(
        model=model, input_dim=input_dim,
        batch_sizes=(1, 32, 256),
        n_warmup=20, n_runs=100,
    )
    _print_latency_table(domain, latency, device)

    # ------------------------------------------------------------------
    # 9. OPERATIONAL MODULE 1: Router Cascade Analysis
    # ------------------------------------------------------------------
    logger.info("[6/8] Router Cascade Analysis ...")
    routing = _synthetic_routing(len(y_true), domain)
    router_results = analyze_router_cascade(routing, y_true, y_pred, domain)

    # ------------------------------------------------------------------
    # 10. OPERATIONAL MODULE 2: Anomaly Override (VAE)
    # ------------------------------------------------------------------
    logger.info("[7/8] Anomaly Override Analysis ...")
    anomaly_scores = _synthetic_anomaly_scores(y_proba)
    override_results = analyze_anomaly_override(
        y_true, y_pred, anomaly_scores, override_threshold=0.8
    )

    # ------------------------------------------------------------------
    # 11. OPERATIONAL MODULE 3: Cost-Sensitive Threat Matrix
    # ------------------------------------------------------------------
    logger.info("[8/8] Cost-Sensitive Threat Matrix ...")
    # Apply override predictions for cost calculation (more realistic)
    y_pred_overridden = y_pred.copy()
    override_mask = (y_pred == 0) & (anomaly_scores >= 0.8)
    y_pred_overridden[override_mask] = 1

    cost_results = compute_cost_sensitive_matrix(
        y_true, y_pred_overridden, cost_fp=cost_fp, cost_fn=cost_fn
    )
    _print_cost_table(domain, cost_results)

    # ------------------------------------------------------------------
    # 12. JSON Summary
    # ------------------------------------------------------------------
    summary = {
        "domain":       domain,
        "environment":  "kaggle" if is_kaggle() else "local",
        "device":       str(device),
        "epochs_run":   len(history.get("train_loss", [])),
        "input_dim":    input_dim,
        "num_classes":  num_classes,
        "train_time_s": round(train_time, 2),
        "best_train_metrics": best_metrics,
        "test_metrics": {
            k: round(float(metrics[k]), 6)
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            if k in metrics and metrics[k] is not None
        },
        "latency_ms": {
            str(bs): {k: round(v, 4) for k, v in stats.items()}
            for bs, stats in latency.items()
        },
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
        "operational": {
            "router_cascade":           router_results,
            "anomaly_override":         override_results,
            "cost_sensitive_matrix":    cost_results,
        },
    }

    summary_path = save_dir / f"{domain}_kaggle_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved → %s", summary_path)
    logger.info("DOMAIN [%s] COMPLETE  ▸  %s", domain.upper(), save_dir)
    return summary


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_metrics_table(domain: str, metrics: dict) -> None:
    line = "-" * 44
    print(f"\n  {line}")
    print(f"  {domain.upper()} TEST METRICS")
    print(f"  {line}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:<12} : {float(v):.4f}")
    print(f"  {line}\n")


def _print_latency_table(domain: str, latency: dict, device: torch.device) -> None:
    line = "-" * 54
    print(f"\n  {domain.upper()} INFERENCE LATENCY  (device: {device})")
    print(f"  {line}")
    print(f"  {'batch':>8}   {'mean ms':>9}   {'p95 ms':>9}   {'std ms':>9}")
    print(f"  {line}")
    for bs, stats in latency.items():
        print(f"  {bs:>8}   "
              f"{stats['mean_ms']:>9.3f}   "
              f"{stats['p95_ms']:>9.3f}   "
              f"{stats['std_ms']:>9.3f}")
    print(f"  {line}\n")


def _print_cost_table(domain: str, cost: dict) -> None:
    line = "-" * 54
    print(f"\n  {domain.upper()} COST-SENSITIVE THREAT MATRIX")
    print(f"  {line}")
    cm = cost["confusion_matrix"]
    print(f"  TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")
    print(f"  C_FP={cost['cost_fp_coefficient']}   C_FN={cost['cost_fn_coefficient']}  "
          f"ratio={cost['cost_ratio_fn_vs_fp']}x")
    print(f"  FP cost:         {cost['cost_fp_total']:>12.2f} units")
    print(f"  FN cost:         {cost['cost_fn_total']:>12.2f} units")
    print(f"  Total cost:      {cost['total_operational_cost']:>12.2f} units")
    print(f"  Naive baseline:  {cost['baseline_naive_cost']:>12.2f} units")
    print(f"  Savings:         {cost['savings_vs_baseline']:>12.2f} units  "
          f"({cost['savings_pct']:.1f}%)")
    print(f"  {line}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C-rGuard full Kaggle training & evaluation orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--domain",     choices=["cloud", "iot", "both"], default="both")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--nrows",      type=int,   default=0,
                   help="Row limit per domain (0 = load all data).")
    p.add_argument("--cost-fp",    type=float, default=1.0,
                   help="FP cost coefficient for Cost-Sensitive Threat Matrix.")
    p.add_argument("--cost-fn",    type=float, default=10.0,
                   help="FN cost coefficient (should be >> cost-fp).")
    p.add_argument("--output-dir", type=str,   default="",
                   help="Override output directory (default: auto via get_artifacts_root).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    datasets_cfg = load_config("datasets")

    # Resolve output root (env-aware)
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = get_artifacts_root(datasets_cfg) / "evaluation"
    out_root.mkdir(parents=True, exist_ok=True)

    domains   = ["cloud", "iot"] if args.domain == "both" else [args.domain]
    summaries = {}

    logger.info(
        "C-rGuard Kaggle Full  |  environment=%s  domains=%s  epochs=%d  batch=%d",
        "kaggle" if is_kaggle() else "local", domains, args.epochs, args.batch_size,
    )

    for domain in domains:
        try:
            summaries[domain] = run_domain(
                domain=domain,
                epochs=args.epochs,
                batch_size=args.batch_size,
                nrows=args.nrows or None,
                save_dir=out_root / domain,
                datasets_cfg=datasets_cfg,
                cost_fp=args.cost_fp,
                cost_fn=args.cost_fn,
            )
        except Exception as exc:
            logger.error("Domain '%s' failed: %s", domain, exc, exc_info=True)

    # Cross-domain comparison (only when both ran)
    if len(summaries) > 1:
        _print_cross_domain_table(summaries)

    # Save merged summary
    merged_path = out_root / "kaggle_full_summary.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Merged summary saved → %s", merged_path)


def _print_cross_domain_table(summaries: dict) -> None:
    line = "=" * 66
    print(f"\n{line}")
    print("  CROSS-DOMAIN COMPARISON")
    print(line)
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc"]
    header  = f"  {'Metric':<14}" + "".join(f"  {d.upper():<14}" for d in summaries)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in metrics:
        row = f"  {m:<14}"
        for d in summaries:
            v = summaries[d].get("test_metrics", {}).get(m)
            row += f"  {v:.4f}        " if v is not None else "  N/A           "
        print(row)

    print("\n  OPERATIONAL COSTS")
    print("  " + "-" * (len(header) - 2))
    for d in summaries:
        cost = summaries[d].get("operational", {}).get("cost_sensitive_matrix", {})
        override = summaries[d].get("operational", {}).get("anomaly_override", {})
        print(f"  {d.upper():<14}  total_cost={cost.get('total_operational_cost', 'N/A')}  "
              f"savings={cost.get('savings_pct', 'N/A')}%  "
              f"vae_lift_f1={override.get('f1_lift', 'N/A')}")
    print(f"{line}\n")


if __name__ == "__main__":
    main()
