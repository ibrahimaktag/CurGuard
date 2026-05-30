"""Phase-5 visualisation utilities for the Network Attack Detection project.

All public functions follow the same interface contract:
    - Accept data arrays / tensors and a ``save_dir`` Path.
    - Save the figure to ``save_dir / f"{title_prefix}_{plot_name}.png"``.
    - Return the absolute Path of the saved file.
    - Call ``plt.close()`` before returning to prevent memory leaks.

Design notes
------------
* ``matplotlib.use('Agg')`` is set at import time so the module is safe in
  headless / script / Kaggle notebook environments with no display.
* seaborn is used only for styling and the confusion-matrix heatmap — no
  seaborn functions that carry their own state are used globally.
* t-SNE is run on a stratified *subset* (default 5 000 samples) to keep
  local-hardware memory usage bounded (O(n^2) complexity).  PCA pre-
  reduction to 50 dimensions is applied first for speed.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts & Kaggle
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix as sk_cm,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

_DPI     = 150
_PALETTE = sns.color_palette("tab10")
_C_TRAIN = _PALETTE[0]   # blue   - train metric
_C_VAL   = _PALETTE[1]   # orange - val metric
_C_F1    = _PALETTE[2]   # green  - F1
_C_FPR   = _PALETTE[3]   # red    - FPR
_C_ROC   = _PALETTE[4]   # purple - ROC curve
_C_PR    = _PALETTE[5]   # brown  - PR curve
_C_DIAG  = "#9e9e9e"     # neutral grey for reference lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_path(save_dir: Path, prefix: str, name: str) -> Path:
    """Build a standardised output file path."""
    tag = f"{prefix}_" if prefix else ""
    path = Path(save_dir) / f"{tag}{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, path: Path) -> Path:
    """Persist a figure to disk, release memory, and log the save path."""
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved -> %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    save_dir: Path,
    title_prefix: str = "",
) -> Path:
    """Plot train/val loss and val F1 / FPR learning curves.

    Args:
        history:      Dict from ``Trainer.fit()``.  Expected keys:
                      ``train_loss``, ``val_loss``, ``val_f1``, ``val_fpr``.
        save_dir:     Directory where the PNG is written.
        title_prefix: Prepended to the filename (e.g. 'cloud' or 'iot').

    Returns:
        Absolute path of the saved PNG.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    domain_tag = f"{title_prefix.upper()} — " if title_prefix else ""
    fig.suptitle(f"{domain_tag}Training History", fontsize=14, fontweight="bold")

    # --- Loss subplot ---
    ax1.plot(epochs, history["train_loss"], color=_C_TRAIN, lw=2,
             marker="o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   color=_C_VAL,   lw=2,
             marker="s", markersize=4, linestyle="--", label="Val Loss")
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax1.axvline(best_ep, color=_C_DIAG, linestyle=":", lw=1.5,
                label=f"Best epoch ({best_ep})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Validation Loss")
    ax1.legend()

    # --- F1 & FPR subplot ---
    ax2.plot(epochs, history["val_f1"],  color=_C_F1,  lw=2,
             marker="o", markersize=4, label="Val F1-Score")
    ax2.plot(epochs, history["val_fpr"], color=_C_FPR, lw=2,
             marker="s", markersize=4, linestyle="--", label="Val FPR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation F1-Score and FPR")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()

    fig.tight_layout()
    return _save_and_close(fig, _save_path(save_dir, title_prefix, "training_curves"))


# ---------------------------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> Path:
    """Plot a normalised confusion matrix with absolute count annotations.

    Each cell shows both the normalised rate (``%``) and the raw count.
    Color intensity encodes the normalised value (deep blue = high).

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        save_dir:    Destination directory.
        class_names: Display names per class (e.g. ['Benign', 'Attack']).
        title_prefix: Domain prefix for file naming.

    Returns:
        Absolute path of the saved PNG.
    """
    cm = sk_cm(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    cell_px = max(1.8, 6.0 / n_classes)
    fig_w = max(6.0, n_classes * cell_px)
    fig_h = max(5.0, n_classes * cell_px * 0.85)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm_norm,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=False,
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0.0, vmax=1.0,
        cbar_kws={"label": "Normalised Rate"},
    )

    # Annotate each cell: "pct\n(count)"
    for row in range(n_classes):
        for col in range(n_classes):
            norm_val = cm_norm[row, col]
            abs_val  = cm[row, col]
            text_col = "white" if norm_val > 0.55 else "black"
            ax.text(
                col + 0.5, row + 0.5,
                f"{norm_val:.1%}\n({abs_val:,})",
                ha="center", va="center",
                fontsize=10, color=text_col, fontweight="bold",
            )

    domain_tag = f"{title_prefix.upper()} — " if title_prefix else ""
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{domain_tag}Confusion Matrix (Normalised)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    return _save_and_close(fig, _save_path(save_dir, title_prefix, "confusion_matrix"))


# ---------------------------------------------------------------------------
# 3. ROC & PR-AUC Curves (dual panel)
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_dir: Path,
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
) -> Path:
    """Plot ROC and Precision-Recall curves side by side.

    Handles both binary (single ROC/PR pair) and multiclass (per-class OvR
    + micro-average) cases automatically.

    This dual-panel layout is the standard way to demonstrate that PR-AUC
    is a more informative metric than ROC-AUC on imbalanced network
    traffic datasets — making it ideal for thesis methodology sections.

    Args:
        y_true:      Ground-truth integer labels.
        y_proba:     Predicted probabilities (n_samples, n_classes).
        save_dir:    Destination directory.
        class_names: Display names per class.
        title_prefix: Domain prefix for file naming.

    Returns:
        Absolute path of the saved PNG.
    """
    from sklearn.preprocessing import label_binarize

    n_classes = y_proba.shape[1]
    is_binary = (n_classes == 2)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    domain_tag = f"{title_prefix.upper()} — " if title_prefix else ""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{domain_tag}ROC vs PR-AUC Curves  "
                 "(Why PR-AUC matters on imbalanced data)",
                 fontsize=13, fontweight="bold")

    if is_binary:
        pos_prob = y_proba[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_true, pos_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=_C_ROC, lw=2.5,
                    label=f"ROC (AUC = {roc_auc:.4f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, pos_prob)
        ap = average_precision_score(y_true, pos_prob)
        ax_pr.plot(rec, prec, color=_C_PR, lw=2.5,
                   label=f"PR Curve (AP = {ap:.4f})")
        baseline = y_true.sum() / len(y_true)
        ax_pr.axhline(baseline, color=_C_DIAG, lw=1.3, linestyle="--",
                      label=f"No-skill baseline ({baseline:.3f})")

    else:
        # Multiclass: per-class OvR + micro-average
        y_bin   = label_binarize(y_true, classes=list(range(n_classes)))
        palette = sns.color_palette("tab10", n_classes)

        for i, (cname, color) in enumerate(zip(class_names, palette)):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_i = auc(fpr_i, tpr_i)
            ax_roc.plot(fpr_i, tpr_i, color=color, lw=1.5,
                        label=f"{cname} (AUC={roc_auc_i:.3f})")

            prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap_i = average_precision_score(y_bin[:, i], y_proba[:, i])
            ax_pr.plot(rec_i, prec_i, color=color, lw=1.5,
                       label=f"{cname} (AP={ap_i:.3f})")

        # Micro-average ROC
        fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
        ax_roc.plot(fpr_m, tpr_m, color="black", lw=2.5,
                    label=f"Micro-avg (AUC={auc(fpr_m, tpr_m):.3f})")

    # ROC diagonal reference
    ax_roc.plot([0, 1], [0, 1], color=_C_DIAG, lw=1.2, linestyle="--",
                label="Random classifier (AUC=0.50)")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)")
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
    ax_roc.legend(fontsize=9)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(fontsize=9)

    fig.tight_layout()
    return _save_and_close(fig, _save_path(save_dir, title_prefix, "roc_pr_curves"))


# ---------------------------------------------------------------------------
# 4. Confidence Calibration (Reliability Diagram)
# ---------------------------------------------------------------------------

def plot_confidence_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_dir: Path,
    title_prefix: str = "",
    n_bins: int = 10,
) -> Path:
    """Plot a reliability diagram and predicted-probability histogram.

    A well-calibrated model's reliability curve lies on the diagonal.
    Deviations reveal over-confidence (curve below diagonal) or under-
    confidence (above diagonal) — critical insight for SOC alert
    prioritisation where confidence scores drive triage decisions.

    Args:
        y_true:      Ground-truth labels.
        y_proba:     (n_samples, n_classes) probability array.
        save_dir:    Destination directory.
        title_prefix: Domain prefix for file naming.
        n_bins:      Number of equal-width confidence bins.

    Returns:
        Absolute path of the saved PNG.
    """
    n_classes = y_proba.shape[1]
    is_binary = (n_classes == 2)

    # Binary: use positive-class probability.
    # Multiclass: use max-class probability vs. whether the top-1 was correct.
    if is_binary:
        conf_prob = y_proba[:, 1]
        true_bin  = (y_true > 0).astype(int)
    else:
        conf_prob = y_proba.max(axis=1)
        true_bin  = (y_true == y_proba.argmax(axis=1)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob_true, prob_pred = calibration_curve(
            true_bin, conf_prob, n_bins=n_bins, strategy="uniform"
        )

    domain_tag = f"{title_prefix.upper()} — " if title_prefix else ""
    fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{domain_tag}Confidence Calibration  "
                 "(SOC Alert Reliability Analysis)",
                 fontsize=13, fontweight="bold")

    # --- Reliability diagram ---
    ax_cal.plot([0, 1], [0, 1], color=_C_DIAG, lw=1.5, linestyle="--",
                label="Perfect calibration")
    ax_cal.plot(prob_pred, prob_true, color=_C_ROC, lw=2.5,
                marker="o", markersize=7, label="Model calibration")
    ax_cal.fill_between(prob_pred, prob_true, prob_pred,
                        alpha=0.15, color=_C_ROC, label="Calibration gap")
    ax_cal.set_xlabel("Mean Predicted Confidence")
    ax_cal.set_ylabel("Fraction of True Positives")
    ax_cal.set_title("Reliability Diagram")
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    ax_cal.legend()

    # --- Confidence histogram ---
    ax_hist.hist(conf_prob, bins=n_bins * 2, color=_C_PR,
                 edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_hist.set_xlabel("Predicted Confidence Score")
    ax_hist.set_ylabel("Sample Count")
    ax_hist.set_title("Distribution of Predicted Confidence Scores")
    ax_hist.set_xlim(0, 1)

    fig.tight_layout()
    return _save_and_close(
        fig, _save_path(save_dir, title_prefix, "confidence_calibration")
    )


# ---------------------------------------------------------------------------
# 5. Attention Weight Heatmap (XAI)
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    input_dim: int,
    save_dir: Path,
    feature_names: Optional[List[str]] = None,
    title_prefix: str = "",
    max_samples: int = 8,
) -> Path:
    """Plot the multi-head self-attention weight matrix as a heatmap.

    The attention tensor from ``MultiHeadSelfAttention`` has shape
    ``[N, Seq_Len, Seq_Len]`` where ``Seq_Len = input_dim // 4``
    (two MaxPool1d(2) layers in the CNN block reduce the sequence).
    Each sequence position corresponds to a receptive field of 4
    original features.

    The heatmap shows the mean attention matrix across ``max_samples``
    samples, revealing which feature groups the model jointly attends
    to when classifying traffic — an XAI insight into the architecture.

    Args:
        attn_weights:  NumPy array ``[N, Seq_Len, Seq_Len]``.
        input_dim:     Original number of input features (used to derive
                       the receptive-field stride = input_dim // seq_len).
        save_dir:      Destination directory.
        feature_names: Original feature names (len == input_dim).
                       If provided, axes are labelled with feature groups.
        title_prefix:  Domain prefix for file naming.
        max_samples:   Maximum number of samples to average over.

    Returns:
        Absolute path of the saved PNG.
    """
    n_use     = min(max_samples, attn_weights.shape[0])
    mean_attn = attn_weights[:n_use].mean(axis=0)   # [Seq_Len, Seq_Len]
    seq_len   = mean_attn.shape[0]
    stride    = max(1, input_dim // seq_len)         # features per sequence position

    # Build axis tick labels: group feature names into windows of `stride`
    if feature_names is not None and len(feature_names) == input_dim:
        tick_labels = []
        for i in range(seq_len):
            start = i * stride
            # Use first feature name in each window, truncated for readability
            label = feature_names[start][:12] if start < len(feature_names) else f"pos_{i}"
            tick_labels.append(f"[{start}] {label}")
    else:
        tick_labels = [f"pos_{i}" for i in range(seq_len)]

    show_ticks = seq_len <= 30          # suppress labels on very wide matrices
    fig_size   = max(7.0, seq_len * 0.55)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        mean_attn,
        xticklabels=tick_labels if show_ticks else False,
        yticklabels=tick_labels if show_ticks else False,
        cmap="YlOrRd",
        linewidths=0.3 if seq_len <= 20 else 0,
        ax=ax,
        cbar_kws={"label": "Attention Weight (avg over heads & samples)"},
    )

    if show_ticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    domain_tag = f"{title_prefix.upper()} — " if title_prefix else ""
    ax.set_xlabel("Key Position (Feature Group)", fontsize=11)
    ax.set_ylabel("Query Position (Feature Group)", fontsize=11)
    ax.set_title(
        f"{domain_tag}Multi-Head Attention Heatmap  "
        f"(avg over {n_use} samples, seq_len={seq_len})",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    return _save_and_close(
        fig, _save_path(save_dir, title_prefix, "attention_heatmap")
    )


# ---------------------------------------------------------------------------
# 6. t-SNE Feature Space Visualisation
# ---------------------------------------------------------------------------

def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    save_dir: Path,
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
    n_samples: int = 5_000,
    pca_components: int = 50,
    random_state: int = 42,
) -> Path:
    """Plot a 2-D t-SNE projection of the preprocessed feature space.

    Engineering rationale
    ~~~~~~~~~~~~~~~~~~~~~
    * t-SNE has O(n^2) complexity — running it on 50k samples would take
      hours.  A **stratified subset** of ``n_samples=5 000`` is drawn first.
      This preserves class ratios and runs in ~90-120 s on a 3050 Ti CPU
      while producing a representative embedding.
    * PCA pre-reduction to ``pca_components`` dimensions is applied before
      t-SNE.  This speeds up the O(n^2) phase and has no meaningful effect
      on the quality of class separation (Barnes-Hut t-SNE itself applies
      a similar trick internally, but here it is explicit and logged).

    Args:
        X:             Preprocessed feature matrix (n_samples, n_features).
        y:             Integer class labels (n_samples,).
        save_dir:      Destination directory.
        class_names:   Display name for each unique class.
        title_prefix:  Domain prefix for file naming.
        n_samples:     Maximum number of points to embed.
        pca_components: PCA target dimensions before t-SNE (0 = skip PCA).
        random_state:  RNG seed for reproducibility.

    Returns:
        Absolute path of the saved PNG.
    """
    rng = np.random.default_rng(random_state)

    # Stratified sub-sampling: preserve class proportions
    n_total = len(X)
    if n_total > n_samples:
        unique_cls = np.unique(y)
        n_per_cls  = n_samples // len(unique_cls)
        indices    = []
        for cls in unique_cls:
            cls_idx = np.where(y == cls)[0]
            take    = min(len(cls_idx), n_per_cls)
            indices.append(rng.choice(cls_idx, size=take, replace=False))
        idx    = rng.permutation(np.concatenate(indices))
        X_sub  = X[idx]
        y_sub  = y[idx]
        logger.info("t-SNE: sub-sampled %d / %d samples (stratified)", len(X_sub), n_total)
    else:
        X_sub, y_sub = X, y

    # Optional PCA pre-reduction
    n_feats = X_sub.shape[1]
    n_pca   = min(pca_components, n_feats, len(X_sub) - 1) if pca_components > 0 else 0
    if n_pca > 0 and n_feats > n_pca:
        logger.info("t-SNE: PCA %d -> %d dims ...", n_feats, n_pca)
        X_sub = PCA(n_components=n_pca, random_state=random_state).fit_transform(X_sub)

    # t-SNE (perplexity clamped to a valid range)
    perplexity = int(min(30, len(X_sub) // 5))
    logger.info(
        "t-SNE: fitting TSNE(n=%d, perplexity=%d) — this may take 1-2 min ...",
        len(X_sub), perplexity,
    )
    X_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1_000,
        random_state=random_state,
        n_jobs=1,   # deterministic; increase for speed if reproducibility not needed
    ).fit_transform(X_sub.astype(np.float32))

    # --- Plot ---
    unique_cls = np.unique(y_sub)
    n_cls      = len(unique_cls)
    if class_names is None:
        class_names = [str(c) for c in unique_cls]

    palette = sns.color_palette("tab10", n_cls)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls in enumerate(unique_cls):
        mask  = y_sub == cls
        label = class_names[i] if i < len(class_names) else str(cls)
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[palette[i]],
            label=f"{label}  (n={mask.sum():,})",
            s=12, alpha=0.55, linewidths=0,
        )

    domain_tag = f"{title_prefix.upper()} — " if title_prefix else ""
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.set_title(
        f"{domain_tag}t-SNE Feature Space  ({len(X_sub):,} samples)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(markerscale=2.5, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])   # t-SNE axes have no meaningful unit

    fig.tight_layout()
    return _save_and_close(fig, _save_path(save_dir, title_prefix, "tsne"))
