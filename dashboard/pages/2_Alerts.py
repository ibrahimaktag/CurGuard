"""Streamlit page: alert thresholds from ensemble config."""

from __future__ import annotations

import streamlit as st

from src.utils.config import load_config


def main() -> None:
    """Display configurable alert thresholds."""
    st.title("Alerts & thresholds")
    cfg = load_config("ensemble")
    alert = cfg.get("alert_thresholds", {})
    anom = cfg.get("anomaly_integration", {})

    st.subheader("Uyarı eşikleri (`configs/ensemble.yaml`)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🚨 alert_thresholds")
        rows1 = []
        for k, v in alert.items():
            rows1.append(f"| **{k}** | `{v}` |")
        st.markdown(
            "| Eşik Tanımı | Değer |\n| :--- | :--- |\n" + "\n".join(rows1)
        )

    with col2:
        st.markdown("#### 🛡️ anomaly_integration")
        rows2 = []
        for k, v in anom.items():
            rows2.append(f"| **{k}** | `{v}` |")
        st.markdown(
            "| Parametre | Değer |\n| :--- | :--- |\n" + "\n".join(rows2)
        )

    st.markdown(
        """
        - **attack_confidence**: Sınıflandırıcı olasılığı bu değerin üzerindeyse saldırı alarmı.
        - **anomaly_score**: Hibrit anomali skoru bu değerin üzerindeyse anomali uyarısı.
        - **low_confidence_warn**: Router güveni bu değerin altındaysa düşük güven uyarısı.
        - **override_threshold**: Anomali skoru bunun üzerindeyse (ve `mode: override`) uzman çıktısı geçersiz sayılır.
        """
    )


main()
