"""Phase-5 Local Evaluation Orchestrator.

Runs a mini training session (default: 5 epochs, 50 000 rows) for a single
domain (cloud, iot, or both), generates the full Phase-5 evaluation suite
of plots, and measures inference latency.  Designed to validate the complete
evaluation pipeline on a local 3050 Ti GPU before scaling to Kaggle.

All generated artefacts (plots + JSON summary) land under::

    artifacts/evaluation/{domain}/

Usage
-----
    # Cloud domain only (default: 5 epochs, 50k rows)
    python scripts/run_phase5_local.py --domain cloud

    # IoT domain, custom settings
    python scripts/run_phase5_local.py --domain iot --epochs 3 --nrows 30000

    # Both domains sequentially
    python scripts/run_phase5_local.py --domain both --epochs 5

    # Dry-run with mock data (no CSV files required)
    python scripts/run_phase5_local.py --domain cloud --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is importable when invoked as `python scripts/run_...`
_PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

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
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain configuration table
# ---------------------------------------------------------------------------
_DOMAIN_CFG = {
    "cloud": {
        "model_cls":    CloudSpecialist,
        "preproc_cls":  CloudPreprocessor,
        "target_col":   "Label",
        "class_names":  ["Benign", "Attack"],
        "task_type":    "binary",
    },
    "iot": {
        "model_cls":    IoTSpecialist,
        "preproc_cls":  IoTPreprocessor,
        "target_col":   "label",
        "class_names":  ["Benign", "Attack"],
        "task_type":    "binary",
    },
}


# ---------------------------------------------------------------------------
# Helper: collect predictions + attention weights via forward hook
# ---------------------------------------------------------------------------

def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task_type: str = "binary",
):
    """Evaluate the model on a DataLoader and gather all outputs.

    A forward hook is registered on ``model.attention.mha`` to capture the
    multi-head attention weight matrix ``[Batch, Seq_Len, Seq_Len]`` for
    every batch, without modifying the model architecture.

    Args:
        model:     Trained PyTorch specialist model.
        loader:    DataLoader wrapping the evaluation set.
        device:    Compute device (CPU or CUDA).
        task_type: ``'binary'`` or ``'multiclass'``.

    Returns:
        y_true   : (n,) integer ground-truth labels.
        y_pred   : (n,) integer predicted labels.
        y_proba  : (n, n_classes) probability array.
        attn_w   : (n, seq_len, seq_len) concatenated attention weights,
                   or ``None`` if weights were not available.
    """
    model.eval()
    all_preds, all_targets, all_probs, all_attn = [], [], [], []

    # --- Attention weight hook ---
    attn_cache: dict = {}

    def _hook(module, inp, out):
        # nn.MultiheadAttention forward returns (attn_output, attn_weights).
        # attn_weights may be None if PyTorch uses fast path — but we forced
        # need_weights=True in MultiHeadSelfAttention.forward(), so it won't be.
        if out[1] is not None:
            attn_cache["w"] = out[1].detach().cpu().numpy()

    handle = model.attention.mha.register_forward_hook(_hook)

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            logits  = model(batch_X)           # [Batch] (binary) or [Batch, n_cls]

            if task_type == "binary":
                probs_pos = torch.sigmoid(logits).cpu().numpy()   # (batch,)
                preds     = (probs_pos > 0.5).astype(np.int64)
                probs_2d  = np.column_stack([1.0 - probs_pos, probs_pos])
            else:
                probs_2d  = torch.softmax(logits, dim=1).cpu().numpy()
                preds     = probs_2d.argmax(axis=1).astype(np.int64)

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

    logger.info(
        "Evaluation complete: n=%d  classes=%s  attn_collected=%s",
        len(y_true), np.unique(y_true).tolist(), attn_w is not None,
    )
    return y_true, y_pred, y_proba, attn_w


# ---------------------------------------------------------------------------
# Helper: compute binary class weights for BCEWithLogitsLoss pos_weight
# ---------------------------------------------------------------------------

def _pos_weight(df: "pd.DataFrame", target_col: str, device: torch.device):
    """Compute pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    counts = df[target_col].value_counts()
    n_neg  = int(counts.get(0, 1))
    n_pos  = int(counts.get(1, 1))
    ratio  = n_neg / max(n_pos, 1)
    logger.info("Class balance: neg=%d  pos=%d  pos_weight=%.3f", n_neg, n_pos, ratio)
    return torch.tensor([ratio], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Helper: build a mock DataFrame (dry-run mode, no real CSVs needed)
# ---------------------------------------------------------------------------

def _make_mock_preprocessed(
    n_rows: int,
    n_feats: int,
    target_col: str,
    seed: int = 0,
) -> "pd.DataFrame":
    """Return a mock DataFrame that looks like it has already been preprocessed.

    Labels are perfectly balanced 0 / 1 floats so that binary DataLoaders
    and metric functions receive both classes without triggering any
    LabelEncoder mismatch inside the specialist preprocessors.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n_rows, n_feats)).astype(np.float32)
    # Perfectly alternating labels to guarantee both classes are present
    y   = (np.arange(n_rows) % 2).astype(np.float32)
    rng.shuffle(y)
    df  = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_feats)])
    df[target_col] = y
    return df


# ---------------------------------------------------------------------------
# Main pipeline for a single domain
# ---------------------------------------------------------------------------

def run_domain(
    domain: str,
    nrows: int,
    epochs: int,
    output_root: Path,
    dry_run: bool = False,
) -> dict:
    """Execute the full Phase-5 pipeline for one domain.

    Steps
    -----
    1. Load & preprocess data  (fit_transform + save stratified CSV sample)
    2. Build DataLoaders
    3. Initialise model and Trainer
    4. Mini training  (``epochs`` epochs)
    5. Plot training curves
    6. Evaluate on test set  (predictions + attention weights via hook)
    7. Compute classification metrics
    8. Generate evaluation plots  (confusion matrix, ROC/PR, calibration,
       attention heatmap, t-SNE)
    9. Measure inference latency  (batch=1 and batch=32)
    10. Save JSON summary

    Args:
        domain:      'cloud' or 'iot'.
        nrows:       Rows to load per domain.
        epochs:      Training epochs for the local mini-run.
        output_root: Parent directory for plots / summaries.
        dry_run:     If True, skip CSV loading and use synthetic mock data.

    Returns:
        Summary dict with key metrics and file paths.
    """
    import pandas as pd

    cfg        = _DOMAIN_CFG[domain]
    save_dir   = output_root / domain
    save_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_col = cfg["target_col"]
    task_type  = cfg["task_type"]
    class_names = cfg["class_names"]

    logger.info("=" * 62)
    logger.info("PHASE 5  |  domain=%-6s  device=%s  nrows=%d  epochs=%d%s",
                domain.upper(), device, nrows, epochs,
                "  [DRY-RUN]" if dry_run else "")
    logger.info("=" * 62)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    logger.info("[1/6] Loading data ...")
    if dry_run:
        # Bypass the full preprocessing pipeline — use pre-encoded mock DataFrames
        # so that integer 0/1 labels never clash with the string-label encode_target
        # implementations inside CloudPreprocessor / IoTPreprocessor.
        logger.warning(
            "DRY-RUN: skipping CSV load and preprocessing; "
            "using balanced mock DataFrames (no CSV files required)."
        )
        n_feats  = 76 if domain == "cloud" else 46
        n_train  = int(nrows * 0.80)
        n_val    = int(nrows * 0.10)
        n_test   = nrows - n_train - n_val
        df_train_c = _make_mock_preprocessed(n_train, n_feats, target_col, seed=0)
        df_val_c   = _make_mock_preprocessed(n_val,   n_feats, target_col, seed=1)
        df_test_c  = _make_mock_preprocessed(n_test,  n_feats, target_col, seed=2)
        feature_names = [f"feat_{i}" for i in range(n_feats)]
        input_dim     = n_feats
        num_classes   = 2
        logger.info(
            "[1/6] Mock data ready: train=%d  val=%d  test=%d  input_dim=%d",
            n_train, n_val, n_test, input_dim,
        )
    else:
        df_raw = load_dataset_raw(domain=domain, nrows=nrows)

        logger.info("[1/6] Splitting (80 / 10 / 10) ...")
        df_train, df_val, df_test = split_train_val_test_df(
            df_raw,
            target_column=target_col,
            test_size=0.10,
            val_size=0.10,
            random_state=42,
        )

        preproc = cfg["preproc_cls"](target_col=target_col, task_type=task_type)

        logger.info("[1/6] Preprocessing — fit_transform (train) + save sample ...")
        df_train_c = preproc.fit_transform(df_train, save_sample=True, n_sample_rows=50)
        df_val_c   = preproc.transform(df_val)
        df_test_c  = preproc.transform(df_test)

        feature_names = [c for c in df_train_c.columns if c != target_col]
        input_dim     = len(feature_names)
        num_classes   = int(df_train_c[target_col].nunique())

        logger.info(
            "[1/6] input_dim=%d  num_classes=%d", input_dim, num_classes
        )

    # ------------------------------------------------------------------
    # 2. DataLoaders
    # ------------------------------------------------------------------
    BATCH_SIZE   = 256
    train_loader = create_dataloader(df_train_c, target_col, task_type,
                                     batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = create_dataloader(df_val_c,   target_col, task_type,
                                     batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = create_dataloader(df_test_c,  target_col, task_type,
                                     batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 3. Model + Trainer
    # ------------------------------------------------------------------
    logger.info("[2/6] Initialising model ...")
    model = cfg["model_cls"](input_dim=input_dim, num_classes=num_classes).to(device)
    model.count_parameters()

    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    class_weight = _pos_weight(df_train_c, target_col, device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        task_type=task_type,
        class_weights=class_weight,
        device=device,
        patience=epochs + 1,           # effectively disable early stopping for local runs
        model_save_path=str(save_dir / f"best_{domain}_model.pt"),
    )

    # ------------------------------------------------------------------
    # 4. Mini training
    # ------------------------------------------------------------------
    logger.info("[3/6] Training for %d epoch(s) ...", epochs)
    t_train_start = time.perf_counter()
    best_metrics, history = trainer.fit(epochs=epochs)
    train_time = time.perf_counter() - t_train_start

    logger.info(
        "Training done in %.1f s  —  best val_f1=%.4f  val_loss=%.4f",
        train_time,
        best_metrics.get("val_f1", 0.0),
        best_metrics.get("val_loss", 0.0),
    )

    # Training curve plot
    logger.info("[3/6] Saving training curves ...")
    plot_training_curves(history, save_dir=save_dir, title_prefix=domain)

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("[4/6] Running inference on test set ...")
    y_true, y_pred, y_proba, attn_weights = collect_predictions(
        model, test_loader, device, task_type
    )

    metrics = compute_classification_metrics(
        y_true, y_pred,
        y_proba=y_proba,
        class_names=class_names,
        average="weighted",
    )

    # Print concise results table
    _print_metrics_table(domain, metrics)

    # ------------------------------------------------------------------
    # 6. Generate evaluation plots
    # ------------------------------------------------------------------
    logger.info("[5/6] Generating evaluation plots ...")

    plot_confusion_matrix(
        y_true, y_pred, save_dir=save_dir,
        class_names=class_names, title_prefix=domain,
    )
    plot_roc_pr_curves(
        y_true, y_proba, save_dir=save_dir,
        class_names=class_names, title_prefix=domain,
    )
    plot_confidence_calibration(
        y_true, y_proba, save_dir=save_dir, title_prefix=domain,
    )

    if attn_weights is not None:
        plot_attention_heatmap(
            attn_weights=attn_weights,
            input_dim=input_dim,
            save_dir=save_dir,
            feature_names=feature_names,
            title_prefix=domain,
            max_samples=16,
        )
    else:
        logger.warning("Attention weights not captured — skipping heatmap.")

    # t-SNE: subset of test features (5 000 samples max, stratified)
    logger.info("[5/6] Running t-SNE on test features (n_samples=5000, PCA=50 dims) ...")
    X_test_np = df_test_c.drop(columns=[target_col]).values.astype(np.float32)
    plot_tsne(
        X=X_test_np, y=y_true,
        save_dir=save_dir,
        class_names=class_names,
        title_prefix=domain,
        n_samples=5_000,
        pca_components=50,
    )

    # ------------------------------------------------------------------
    # 7. Inference latency measurement
    # ------------------------------------------------------------------
    logger.info("[6/6] Measuring inference latency (batch=1 and batch=32) ...")
    latency = measure_inference_latency_torch(
        model=model,
        input_dim=input_dim,
        batch_sizes=(1, 32),
        n_warmup=20,
        n_runs=100,
    )
    _print_latency_table(domain, latency, device)

    # ------------------------------------------------------------------
    # 8. Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "domain":        domain,
        "nrows_loaded":  nrows,
        "epochs_run":    len(history["train_loss"]),
        "input_dim":     input_dim,
        "num_classes":   num_classes,
        "task_type":     task_type,
        "device":        str(device),
        "train_time_s":  round(train_time, 2),
        "best_train_metrics": best_metrics,
        "test_metrics": {
            k: (round(float(metrics[k]), 6) if metrics[k] is not None else None)
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
            if k in metrics
        },
        "latency_ms": {
            str(bs): {k: round(v, 4) for k, v in stats.items()}
            for bs, stats in latency.items()
        },
        "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
    }

    summary_path = save_dir / f"{domain}_phase5_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON saved -> %s", summary_path)

    logger.info(
        "PHASE 5 [%s] COMPLETE  |  output_dir=%s", domain.upper(), save_dir
    )
    return summary


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_metrics_table(domain: str, metrics: dict) -> None:
    """Print a concise metric table to stdout."""
    line = "-" * 44
    print(f"\n  {line}")
    print(f"  {domain.upper()} TEST METRICS")
    print(f"  {line}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:<12} : {float(v):.4f}")
    print(f"  {line}\n")


def _print_latency_table(
    domain: str, latency: dict, device: torch.device
) -> None:
    """Print inference latency table to stdout."""
    line = "-" * 50
    print(f"\n  {domain.upper()} INFERENCE LATENCY  (device: {device})")
    print(f"  {line}")
    print(f"  {'batch':>6}   {'mean ms':>9}   {'p95 ms':>9}   {'std ms':>9}")
    print(f"  {line}")
    for bs, stats in latency.items():
        print(
            f"  {bs:>6}   "
            f"{stats['mean_ms']:>9.3f}   "
            f"{stats['p95_ms']:>9.3f}   "
            f"{stats['std_ms']:>9.3f}"
        )
    print(f"  {line}\n")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-5 local evaluation orchestrator (50k subset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--domain",
        choices=["cloud", "iot", "both"],
        default="cloud",
        help="Domain to evaluate.",
    )
    p.add_argument(
        "--nrows", type=int, default=50_000,
        help="Number of rows to load per domain.",
    )
    p.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs for the local mini-run.",
    )
    p.add_argument(
        "--output-dir", type=str, default="artifacts/evaluation",
        help="Root directory for plots and JSON summaries.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic mock data — no real CSV files required.  "
             "Useful for pipeline/plot testing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args      = parse_args()
    out_root  = _PROJECT_ROOT / args.output_dir
    domains   = ["cloud", "iot"] if args.domain == "both" else [args.domain]
    summaries = {}

    for domain in domains:
        try:
            summaries[domain] = run_domain(
                domain=domain,
                nrows=args.nrows,
                epochs=args.epochs,
                output_root=out_root,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            logger.error(
                "Domain '%s' failed: %s", domain, exc, exc_info=True
            )

    # Cross-domain comparison table (shown when both domains ran)
    if len(summaries) > 1:
        print("\n" + "=" * 58)
        print("  CROSS-DOMAIN COMPARISON")
        print("=" * 58)
        fmt_h = f"  {'Metric':<14}" + "".join(
            f"  {d.upper():<12}" for d in summaries
        )
        print(fmt_h)
        print("  " + "-" * (len(fmt_h) - 2))
        for metric in ["accuracy", "f1", "roc_auc", "pr_auc"]:
            row = f"  {metric:<14}"
            for d in summaries:
                v = summaries[d].get("test_metrics", {}).get(metric)
                row += f"  {v:.4f}      " if v is not None else "  N/A         "
            print(row)
        print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
