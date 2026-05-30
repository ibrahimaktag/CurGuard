# Proje Özeti ve Geliştirme Yol Haritası

Bu dosyanın amacı, **grup üyelerinin** tek bakışta (1) **şu an repoda neyin bittiğini**, her parçanın ne işe yaradığını ve (2) **bitirme tesliminden sonra planlanan ek geliştirmelerin** kapsamını görmesidir.

---

## Bu belge neyi kapsar, neyi kapsamaz?

| Kapsar | Kapsamaz |
|--------|----------|
| Tamamlanmış kodun modül modül özeti | Satır satır implementasyon rehberi |
| Bir sonraki sürüm için **kararlaştırılmış** hedefler (Docker, API, PCAP POC vb.) | Adım adım kurulum/runbook (bunlar yapılırken README veya ayrı dokümana yazılır) |
| Gelecekte eklenecek dosya/klasör şeması (taslak) | O dosyaların içinde yazılacak kodun detayı |
| Fazların önceliği ve kabaca efor | Kesin tarih taahhüdü |

Yani: **yol haritası ve mimari karar özeti**; “nasıl yapacağız?”nın **teknik uygulama kılavuzu** değildir. Uygulama sırasında çıkan detaylar README, PR açıklamaları veya kod içi yorumlarla güncellenir.

---

## Bölüm A — Mevcut proje (kod olarak teslim edilen hali)

Aşağıdaki özet, **bitirme projesi kapsamında tamamlanan** yazılım bileşenlerini anlatır. Veri setlerinin fiziksel olarak repoda olmaması (`data/raw/` genelde `.gitignore`) normaldir; eğitilmiş modeller de `artifacts/` altında yerelde üretilir.

### A.1 Genel akış

Ham trafik **CSV** olarak okunur (CIC-IoT2023 ve CIC-IDS2018 ile uyumlu yapı). Domain ön işleme → **router** (IoT vs Cloud) → ilgili **uzman sınıflandırıcı** → **ensemble** katmanında birleştirme → **iki ayrı anomali modeli** (IoT/Cloud özellik uzayına göre) ile skor. İsteğe bağlı **SHAP** açıklaması ve **Streamlit** arayüzü.

### A.2 Veri katmanı (`src/data/`)

| Bileşen | Dosya / not | Ne işe yarar |
|---------|-------------|--------------|
| CSV yükleme | `loader.py` | Ham klasörden CSV okuma; IoT’ta etiket çoğu zaman dosya adından; Cloud’ta `Label` sütunu; hatalı header satırları elenir. |
| Ön işleme | `base_preprocessor.py`, `iot_preprocessor.py`, `cloud_preprocessor.py` | Sayısal özellik seçimi, ölçekleme (StandardScaler), etiket kodlama; sadece train üzerinde fit. |
| Router | `router.py` | `configs/router.yaml` içindeki `feature_mapping` ile iki veri setinden ortak özellik uzayı; LightGBM ile eğitim, `route(df)` ile tahmin. |
| Mapping doğrulama | `router_mapping_validate.py` | Ham CSV başlıkları ile YAML eşlemesinin tutarlılığını kontrol (CLI: `scripts/validate_router_mapping.py`). |
| Train/val/test bölme | `src/training/splits.py` | Tabular DataFrame için stratify’lı ayrım. |

**Konfigürasyon:** `configs/datasets.yaml` (yollar, hedef sütunu, sınıf listeleri), `configs/router.yaml`.

### A.3 Modeller (`src/models/`)

| Bileşen | Ne ile | Ne işe yarar |
|---------|--------|----------------|
| IoT / Cloud uzmanı | `iot_specialist.py`, `cloud_specialist.py` — Keras: CNN + LSTM + MultiHeadAttention | Çok sınıflı saldırı türü tahmini; config: `configs/iot_model.yaml`, `configs/cloud_model.yaml`. |
| Anomali | `anomaly_detector.py` | VAE + Isolation Forest hibrit skor; yalnızca “normal” örneklerle eğitim. |
| Ensemble | `ensemble.py` | Router çıktısına göre uzman tahminlerini birleştirir; IoT ve Cloud için **ayrı** yüklü anomali modelleri; eski tek `anomaly_detector` klasörü varsa geriye dönük uyum. |

**Konfigürasyon:** `configs/anomaly_detector.yaml`, `configs/ensemble.yaml` (uyarı eşikleri, `anomaly_models` klasör adları).

### A.4 Eğitim (`src/training/`)

| Bileşen | Ne işe yarar |
|---------|----------------|
| `trainer.py` | Keras `fit`, sınıf ağırlığı, callback’ler, MLflow isteğe bağlı. |
| `callbacks.py` | Checkpoint, early stopping (TensorFlow içe aktarır). |
| `__init__.py` | Callback’ler **tembel yüklenir**; `evaluate_router` gibi scriptler TensorFlow yüklemeden `splits` kullanabilir. |

**CLI scriptleri (`scripts/`):** `train_router.py`, `train_iot.py`, `train_cloud.py`, `train_anomaly.py` — IoT/Cloud eğitiminde `preprocessor.joblib` model klasörüne kaydedilir (dashboard/XAI için).

### A.5 Değerlendirme

| Bileşen | Dosya | Ne işe yarar |
|---------|--------|--------------|
| Metrikler | `src/evaluation/metrics.py` | Accuracy, F1, ROC/PR (olasılık varsa), confusion matrix, gecikme ölçümü. |
| SHAP | `src/evaluation/explainer.py` | Gradient/Deep/Kernel explainer ile top-k özellik. |
| Uzman değerlendirme | `scripts/evaluate_specialist.py` | Test split üzerinde metrik + `artifacts/evaluation/*_predictions.csv`. |
| Router değerlendirme | `scripts/evaluate_router.py` | IoT/Cloud test parçalarında domain doğruluğu; `--nrows` dosya başına limit. |

### A.6 Dashboard (`dashboard/`)

Streamlit: `app.py` (artifact özeti), `status.py`, `evaluation_summary.py` (JSON özetleri), `inference.py` (model + preprocessor yükleme). Sayfalar: Monitoring, Alerts, XAI (SHAP), Metrikler (CSV yükleme, confusion matrix).

### A.7 Testler (`tests/`)

`test_training/test_splits.py`, `test_data/test_router_mapping_validate.py`, `test_data/test_router_features.py` — pytest; `pyproject.toml` içinde pytest ayarları.

### A.8 Dokümantasyon (kök)

`README.md` — kurulum, veri yolları, eğitim sırası, dashboard, değerlendirme komutları.

---

## Bölüm B — Sonraki geliştirme aşaması (planlanan, henüz kodlanmamış)

Bu bölüm, **bitirme tesliminden sonra** eklenmesi kararlaştırılan iyileştirmelerin kapsamını özetler. Kod repoda henüz yoksa normaldir.

### B.1 Hedefler (özet)

| Hedef | Açıklama |
|--------|-----------|
| **Tekrarlanabilir ortam** | Docker ile kurulum tutarlılığı. |
| **Programatik kullanım** | REST API ile CSV/JSON gönderip tahmin. |
| **Opsiyonel köprü** | PCAP → örnek flow CSV (POC; tam üretim IDS değil). |
| **Dokümantasyon** | Docker/API kullanımı ve sınırlamalar. |

**Kapsam dışı (bilinçli):** Tam canlı ağ entegrasyonu (Zeek agent, mirror port, çok kiracılı panel).

### B.2 Öncelik sıralı görev listesi

#### Faz A — Dağıtım

1. `Dockerfile` (`python:3.11-slim`, `requirements.txt`, çalışma dizini)
2. `.dockerignore` (`data/raw`, `artifacts`, `venv`, `.git` …)
3. İsteğe bağlı `docker-compose.yml`
4. README’de `docker build` / `docker run` örnekleri

#### Faz B — Inference REST API

1. FastAPI: `GET/POST /health`, tahmin uçları (specialist / ileride ensemble)
2. Minimum güvenlik: boyut limiti, isteğe bağlı API key
3. `pytest` ile smoke test

#### Faz C — PCAP köprüsü (POC)

1. Harici araç notu: `tshark` / `tcpdump`
2. `scripts/pcap_to_flow_csv.py` (veya `tools/`) — çıktının CIC ile birebir aynı olmayabileceği raporda belirtilir
3. README uyarısı

#### Faz D — İnce ayarlar

1. `pip install -e .` ile paketleşme
2. İsteğe bağlı GitHub Actions (pytest/lint)
3. Router eğitiminde train/test sızıntısını azaltan refactor (orta efor)

### B.3 Planlanan dosya / klasör şeması (taslak)

```
AI-Driven-Network-Attack-Detection/
├── docs/
│   └── DEVELOPMENT_ROADMAP.md
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh          # opsiyonel
├── .dockerignore
├── docker-compose.yml         # opsiyonel
├── src/
│   └── api/                   # FastAPI
│       ├── __init__.py
│       ├── main.py
│       ├── routes/
│       └── schemas.py
├── scripts/
│   └── pcap_to_flow_csv.py    # Faz C
├── tests/
│   └── test_api/
├── requirements.txt           # fastapi, uvicorn vb. eklenebilir
└── README.md                  # Docker + API bölümleri güncellenir
```

### B.4 Bağımlılık notu (Faz B)

- `fastapi`, `uvicorn[standard]`, isteğe bağlı `python-multipart`
- İstenirse `requirements-api.txt` ile ana eğitim ortamından ayrılabilir.

### B.5 Kabaca efor

| Faz | Süre (tek geliştirici, kabaca) |
|-----|----------------------------------|
| A Docker | 0.5–1 gün |
| B API | 1–2 gün |
| C PCAP POC | 1–3 gün |
| D Polish | 0.5–1 gün |

### B.6 Bitirme teslimi ile ilişki

- Bölüm B **bitirme zorunluluğu değildir**; tezde “Gelecek çalışmalar” olarak referans verilebilir.
- Bölüm A, **mevcut teslim kapsamı**dır.

### B.7 Sonraki adım (ekip içi)

1. Bölüm B’den hangi fazların yapılacağını sabitlemek (ör. A+B).
2. Uygulama sırasında: Dockerfile → API iskeleti → README güncellemesi → (isteğe bağlı) PCAP script.
