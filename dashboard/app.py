"""Streamlit entry: network attack detection dashboard (ensemble overview)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from dashboard.status import artifact_status


def main() -> None:
    """Render the main dashboard page."""
    st.set_page_config(
        page_title="AI Network Attack Detection",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("AI-Driven Network Attack Detection")
    st.caption("Ensemble: Router → IoT/Cloud specialists → Anomaly (VAE + Isolation Forest)")

    st.subheader("Model artifacts")
    status = artifact_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Router (IoT vs Cloud)", "ready" if status["router"] else "missing")
    c2.metric("IoT specialist", "ready" if status["iot_specialist"] else "missing")
    c3.metric("Cloud specialist", "ready" if status["cloud_specialist"] else "missing")

    a1, a2, a3 = st.columns(3)
    a1.metric("Anomaly (IoT uzayı)", "ready" if status["anomaly_iot"] else "missing")
    a2.metric("Anomaly (Cloud uzayı)", "ready" if status["anomaly_cloud"] else "missing")
    a3.metric("Anomaly (tek model, eski)", "ready" if status["anomaly_legacy"] else "—")

    st.divider()
    st.markdown(
        """
        **Kullanım:** Ham veriyi `data/raw/` altına koyun; eğitim için proje kökünden
        `python scripts/train_router.py`, `train_iot.py`, `train_cloud.py`,
        `train_anomaly.py --domain iot` ve `train_anomaly.py --domain cloud` çalıştırın.

        Soldaki menüde dört panel: **Monitoring**, **Alerts**, **XAI (SHAP)**,
        **Metrikler** (confusion matrix, ROC/PR opsiyonel).
        """
    )


if __name__ == "__main__":
    main()
