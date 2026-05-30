"""Streamlit page: sınıflandırma metrikleri ve confusion matrix."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import compute_classification_metrics


def main() -> None:
    """Render evaluation metrics from uploaded labels or demo data."""
    st.title("Metrikler & confusion matrix")
    st.caption(
        "Değerlendirme CSV: `y_true` ve `y_pred` sütunları (tamsayı sınıf indeksi). "
        "İsteğe bağlı: `proba_0` … `proba_k` sütunları ile ROC/PR-AUC."
    )

    use_demo = st.checkbox("Demo: rastgele küçük örnek göster", value=False)

    if use_demo:
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 3, size=200)
        y_pred = np.clip(y_true + rng.randint(-1, 2, size=200), 0, 2)
        y_proba = None
        class_names = ["A", "B", "C"]
    else:
        uploaded = st.file_uploader("predictions.csv", type=["csv"])
        if uploaded is None:
            st.info("CSV yükleyin veya demo kutusunu işaretleyin.")
            return
        df = pd.read_csv(io.BytesIO(uploaded.read()), low_memory=False)
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            st.error("CSV içinde `y_true` ve `y_pred` sütunları olmalı.")
            return
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        proba_cols = [c for c in df.columns if c.startswith("proba_")]
        if proba_cols:
            proba_cols = sorted(proba_cols, key=lambda x: int(x.split("_")[-1]))
            y_proba = df[proba_cols].to_numpy(dtype=np.float64)
        else:
            y_proba = None
        class_names = st.text_input(
            "Sınıf adları (virgülle, opsiyonel)",
            value="",
            help="Boş bırakılırsa indeks kullanılır.",
        )
        class_names = [x.strip() for x in class_names.split(",") if x.strip()] or None

    metrics = compute_classification_metrics(
        y_true.astype(np.int32),
        y_pred.astype(np.int32),
        y_proba=y_proba,
        class_names=class_names,
        average="weighted",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("F1 (weighted)", f"{metrics['f1']:.4f}")
    c3.metric("ROC-AUC", "—" if metrics.get("roc_auc") is None else f"{metrics['roc_auc']:.4f}")
    c4.metric("PR-AUC", "—" if metrics.get("pr_auc") is None else f"{metrics['pr_auc']:.4f}")

    with st.expander("Per-class F1"):
        st.json(metrics["per_class_f1"])

    st.subheader("Confusion matrix")
    cm = np.array(metrics["confusion_matrix"])
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if class_names and len(class_names) == len(labels):
        tick = class_names
    else:
        tick = [str(i) for i in labels]
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
        x=tick,
        y=tick,
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification report")
    st.text(metrics["classification_report"])


main()
