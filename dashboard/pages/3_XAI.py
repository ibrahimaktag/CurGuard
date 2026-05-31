"""Streamlit page: SHAP tabanlı tahmin açıklaması (top-k özellik)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.inference import load_specialist_bundle
from src.evaluation.explainer import explain_prediction
from src.utils.config import load_config


def main() -> None:
    """Render SHAP explanation UI."""
    st.title("XAI — SHAP açıklaması")
    st.caption(
        "Model tahmin kararlarının ağ özniteliklerine göre açıklanabilirlik analizi (SHAP değerleri)."
    )

    domain = st.radio("Domain", ("iot", "cloud"), horizontal=True)
    top_k = int(load_config("ensemble").get("xai", {}).get("top_k_features", 10))

    model, prep = load_specialist_bundle(domain)
    if model is None:
        st.error(
            f"❌ {domain.upper()} dikeyine ait uzman model dosyası bulunamadı. Lütfen model eğitimi tamamlayıp "
            f"ilgili dosyaları `artifacts/models/{domain}_specialist/` klasörüne eklediğinizden emin olun."
        )
        return

    n_features = int(model.input_shape[1])
    st.success(f"Model yüklendi ({n_features} özellik).")

    use_demo = st.checkbox("Örnek sentetik veri ile demo (CSV gerekmez)", value=prep is None)
    uploaded = st.file_uploader(
        "Ham akış CSV (eğitimle aynı kolonlar + hedef sütunu)",
        type=["csv"],
    )

    if use_demo or prep is None:
        rng = np.random.RandomState(42)
        X_bg = rng.standard_normal((min(100, max(50, n_features)), n_features)).astype(np.float32)
        X_explain = X_bg[:5]
        names = [f"f{i}" for i in range(n_features)]
        with st.spinner("SHAP hesaplanıyor (ilk seferde yavaş olabilir)..."):
            out = explain_prediction(
                model,
                X_explain,
                names,
                X_background=X_bg,
                top_k=top_k,
            )
        st.info("Sentetik veri kullanıldı; özellik adları `f0…` placeholder’dır.")
    else:
        if uploaded is None:
            st.warning("⚠️ Lütfen analiz edilmesini istediğiniz ağ akış verilerini içeren bir CSV dosyası yükleyin veya yukarıdan sentetik demoyu seçin.")
            return
        raw = uploaded.read()
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        try:
            X, _y = prep.transform(df, include_target=True)
        except Exception as e:
            st.error(f"Ön işleme hatası: {e}")
            return
        if X.shape[1] != n_features:
            st.error(f"Beklenen {n_features} özellik, gelen {X.shape[1]}.")
            return
        names = prep.feature_columns
        bg = X[: min(200, len(X))]
        ex = X[: min(20, len(X))]
        with st.spinner("SHAP hesaplanıyor..."):
            out = explain_prediction(
                model,
                ex,
                names,
                X_background=bg,
                top_k=top_k,
            )

    tops = out["top_features"]
    st.subheader(f"Top {len(tops)} özellik (ortalama |SHAP|)")
    chart = pd.DataFrame(tops, columns=["Özellik", "Önem"])
    st.bar_chart(chart.set_index("Özellik"))


main()
