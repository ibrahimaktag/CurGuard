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
    st.write("**alert_thresholds**")
    st.json(alert)
    st.write("**anomaly_integration**")
    st.json(anom)

    st.markdown(
        """
        - **attack_confidence**: Sınıflandırıcı olasılığı bu değerin üzerindeyse saldırı alarmı.
        - **anomaly_score**: Hibrit anomali skoru bu değerin üzerindeyse anomali uyarısı.
        - **low_confidence_warn**: Router güveni bu değerin altındaysa düşük güven uyarısı.
        - **override_threshold**: Anomali skoru bunun üzerindeyse (ve `mode: override`) uzman çıktısı geçersiz sayılır.
        """
    )


main()
