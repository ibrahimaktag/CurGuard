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
    with st.expander("📖 Kullanım Kılavuzu & Komutlar"):
        st.markdown(
            """
            Ham veriyi `data/raw/` dizini altına yerleştirdikten sonra, tüm sistemi sırasıyla aşağıdaki komutlarla eğitebilir ve değerlendirebilirsiniz:

            * **Yönlendirici Eğitimi**:
              `python scripts/train_router.py`
            * **IoT Specialist Eğitimi**:
              `python scripts/train_iot.py`
            * **Cloud Specialist Eğitimi**:
              `python scripts/train_cloud.py`
            * **IoT Anomaly Detector Eğitimi**:
              `python scripts/train_anomaly.py --domain iot`
            * **Cloud Anomaly Detector Eğitimi**:
              `python scripts/train_anomaly.py --domain cloud`
            
            Sol taraftaki menü üzerinden panel geçişlerini kullanabilirsiniz:
            - **Monitoring**: Genel model doğruluk ve gecikme metrikleri özeti.
            - **Alerts**: Konfigüre edilmiş uyarı ve anomali filtre eşikleri.
            - **XAI (SHAP)**: Karar mekanizmalarının açıklanabilirlik analizleri.
            - **Metrics**: Yüklenen tahmin dosyalarına göre hata matrisi (confusion matrix) çizimi.
            """
        )


if __name__ == "__main__":
    main()
