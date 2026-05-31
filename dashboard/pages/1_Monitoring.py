"""Streamlit page: training / deployment monitoring placeholders."""

from __future__ import annotations

import streamlit as st

from dashboard.evaluation_summary import load_all_evaluation_summaries
from dashboard.status import artifact_status


def main() -> None:
    """Show artifact readiness and saved evaluation metrics when available."""
    st.title("Monitoring")
    st.caption("Sistem bileşenlerinin durumları ve test performanslarının genel özeti.")

    status = artifact_status()
    ready = sum(1 for k, v in status.items() if v and k != "anomaly_legacy")
    total = 5
    st.progress(ready / total, text=f"Hazır bileşen: {ready}/{total} (legacy hariç)")

    with st.expander("Bileşen durumu", expanded=True):
        friendly_names = {
            "router": "Router (IoT vs Cloud)",
            "iot_specialist": "IoT Specialist Model",
            "cloud_specialist": "Cloud Specialist Model",
            "anomaly_iot": "Anomaly Detector (IoT)",
            "anomaly_cloud": "Anomaly Detector (Cloud)",
            "anomaly_legacy": "Anomaly Detector (Legacy)"
        }
        rows = []
        for k, v in status.items():
            status_text = "🟢 Hazır (OK)" if v else "🔴 Eksik (Missing)"
            rows.append(f"| **{friendly_names.get(k, k)}** | {status_text} |")
        
        st.markdown(
            "| Bileşen Adı | Mevcut Durum |\n| :--- | :--- |\n" + "\n".join(rows)
        )

    sums = load_all_evaluation_summaries()
    st.subheader("Son değerlendirme özetleri")
    cols = st.columns(3)
    titles = (
        ("IoT specialist", "iot"),
        ("Cloud specialist", "cloud"),
        ("Router", "router"),
    )
    for col, (title, key) in zip(cols, titles):
        data = sums.get(key)
        with col:
            st.markdown(f"**{title}**")
            if data is None:
                st.caption("Veri yok — ilgili evaluate scriptini çalıştırın.")
            else:
                if key == "router":
                    st.metric("Accuracy (strict)", f"{data.get('accuracy_strict', 0):.4f}")
                    st.caption(
                        f"n_iot={data.get('n_test_iot')} · n_cloud={data.get('n_test_cloud')} · "
                        f"ms/örn≈{data.get('inference_ms_per_sample', 0):.3f}"
                    )
                else:
                    st.metric("Accuracy", f"{data.get('accuracy', 0):.4f}")
                    st.metric("F1 (weighted)", f"{data.get('f1_weighted', 0):.4f}")
                    roc = data.get("roc_auc")
                    st.caption(
                        "ROC-AUC: "
                        + ("—" if roc is None else f"{roc:.4f}")
                        + " · ms/örn≈"
                        + f"{data.get('latency_ms', {}).get('mean_ms', 0):.3f}"
                    )

    with st.expander("🛠️ Geliştirici Bilgileri & Değerlendirme Komutları"):
        st.markdown(
            """
            Değerlendirme sonuçlarını ve metrik raporlarını sıfırdan yeniden üretmek için şu komutları kullanabilirsiniz:
            * **Uzman Model Değerlendirme**:
              `python scripts/evaluate_specialist.py --domain iot|cloud`
            * **Yönlendirici Değerlendirme**:
              `python scripts/evaluate_router.py`
            
            Model eğitimi ve test aşamalarındaki tüm hiperparametre takipleri **MLflow** (`artifacts/mlruns`) üzerinde kayıt altındadır.
            """
        )


main()
