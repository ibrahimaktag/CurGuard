"""Streamlit page: training / deployment monitoring placeholders."""

from __future__ import annotations

import streamlit as st

from dashboard.evaluation_summary import load_all_evaluation_summaries
from dashboard.status import artifact_status


def main() -> None:
    """Show artifact readiness and saved evaluation metrics when available."""
    st.title("Monitoring")
    st.caption("`artifacts/evaluation/*.json` dosyaları scriptler çalıştıktan sonra burada özetlenir.")

    status = artifact_status()
    ready = sum(1 for k, v in status.items() if v and k != "anomaly_legacy")
    total = 5
    st.progress(ready / total, text=f"Hazır bileşen: {ready}/{total} (legacy hariç)")

    with st.expander("Bileşen durumu"):
        st.json({k: ("ok" if v else "eksik") for k, v in status.items()})

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

    st.info(
        "Komutlar: `python scripts/evaluate_specialist.py --domain iot|cloud`, "
        "`python scripts/evaluate_router.py`. MLflow: `artifacts/mlruns`."
    )


main()
