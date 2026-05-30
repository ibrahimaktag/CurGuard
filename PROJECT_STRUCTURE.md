# 📁 C-rGuard — Proje Klasör Yapısı

> **Repo:** `ibrahimaktag/C-rGuard`  
> **Dil:** Python 3.x  
> **Amaç:** AI tabanlı Ağ Saldırısı Tespit Sistemi (Network Intrusion Detection)

---

## 🗂️ Klasör Ağacı

```
C-rGuard/
│
├── README.md                           # Proje açıklaması ve kullanım kılavuzu
├── PROJECT_STRUCTURE.md                # Bu dosya — klasör yapısı şablonu
├── requirements.txt                    # Python bağımlılıkları
├── pyproject.toml                      # Proje metadata ve build ayarları
├── pyrightconfig.json                  # Pyright (tip kontrolü) yapılandırması
├── .gitignore                          # Git'e dahil edilmeyecek dosyalar
├── mlflow.db                           # MLflow deney takip veritabanı
│
├── configs/                            # Tüm model ve sistem YAML konfigürasyonları
│   ├── anomaly_detector.yaml           # Anomali dedektörü hiper-parametreleri
│   ├── cloud_model.yaml                # Cloud (CICIDS-2018) model ayarları
│   ├── datasets.yaml                   # Dataset yolları + ortam (local/Kaggle) ayarları
│   ├── eda_samples.yaml                # EDA örnek oluşturma ayarları
│   ├── ensemble.yaml                   # Ensemble model ayarları
│   ├── iot_model.yaml                  # IoT (CICIOT-2023) model ayarları
│   ├── router.yaml                     # Trafik yönlendirici ayarları
│   └── training.yaml                   # Genel eğitim parametreleri
│
├── src/                                # Ana kaynak kodu
│   ├── __init__.py
│   │
│   ├── data/                           # Veri yükleme ve ön işleme
│   │   ├── __init__.py
│   │   ├── base_preprocessor.py        # Ortak ön işleme sınıfı (base class)
│   │   ├── cloud_preprocessor.py       # Cloud trafiği özelleştirilmiş ön işleme
│   │   ├── iot_preprocessor.py         # IoT trafiği özelleştirilmiş ön işleme
│   │   ├── loader.py                   # Ham veri yükleme ve parçalama (chunked)
│   │   ├── preprocessing.py            # Genel ön işleme pipeline'ı
│   │   ├── router.py                   # Trafik tipi yönlendirici (cloud/iot)
│   │   └── router_mapping_validate.py  # Router feature mapping doğrulaması
│   │
│   ├── models/                         # Model mimarileri
│   │   ├── __init__.py
│   │   ├── base_model.py               # Temel model sınıfı
│   │   ├── attention.py                # Self-Attention mekanizması
│   │   ├── anomaly_detector.py         # Anomali tespit modeli (Autoencoder/VAE)
│   │   ├── cloud_specialist.py         # Cloud trafiği uzman modeli
│   │   ├── iot_specialist.py           # IoT trafiği uzman modeli
│   │   └── ensemble.py                 # Ensemble (Router + Specialist) sistemi
│   │
│   ├── training/                       # Model eğitim kodları
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Genel trainer sınıfı (early stopping vb.)
│   │   ├── train_model.py              # Model eğitim giriş noktası
│   │   └── splits.py                   # Train/val/test bölme işlemleri
│   │
│   ├── evaluation/                     # Model değerlendirme
│   │   ├── __init__.py
│   │   ├── metrics.py                  # F1, AUC, Precision, Recall + latency ölçümü
│   │   ├── plots.py                    # ROC, PR, Confusion Matrix, t-SNE görselleştirme
│   │   └── explainer.py                # XAI (SHAP/Attention) açıklanabilirlik
│   │
│   └── utils/                          # Yardımcı araçlar
│       ├── __init__.py
│       ├── config.py                   # Ortam-bilinçli path yönetimi (Local & Kaggle)
│       ├── logger.py                   # Loglama sistemi
│       └── seed.py                     # Reproducibility için random seed
│
├── dashboard/                          # Streamlit Web Dashboard
│   ├── __init__.py
│   ├── app.py                          # Dashboard ana giriş noktası
│   ├── inference.py                    # Gerçek zamanlı tahmin sayfası
│   ├── evaluation_summary.py           # Değerlendirme özet sayfası
│   ├── status.py                       # Sistem durum sayfası
│   └── pages/                          # Çok sayfalı Streamlit yapısı
│       ├── __init__.py
│       ├── 1_Monitoring.py             # Sayfa 1: Canlı trafik izleme
│       ├── 2_Alerts.py                 # Sayfa 2: Saldırı uyarıları
│       ├── 3_XAI.py                    # Sayfa 3: Açıklanabilir AI (SHAP/Attention)
│       └── 4_Metrics.py                # Sayfa 4: Model performans metrikleri
│
├── scripts/                            # Çalıştırılabilir yardımcı scriptler
│   │
│   ├── ── Eğitim Scriptleri ──────────────────────────────────
│   ├── train_router.py                 # Router modelini eğitme
│   ├── train_cloud.py                  # Cloud modelini eğitme
│   ├── train_iot.py                    # IoT modelini eğitme
│   ├── train_anomaly.py                # Anomali dedektörünü eğitme
│   │
│   ├── ── Pipeline Orchestrators ─────────────────────────────
│   ├── run_phase5_local.py             # Faz 5 tam pipeline (local ortam)
│   ├── run_kaggle_full.py              # ★ YENİ — Kaggle GPU ortamı için tam pipeline
│   │                                  #   Eğitim + değerlendirme + 3 operasyonel modül:
│   │                                  #   1) Router Cascade Analysis
│   │                                  #   2) Anomaly Override Module (VAE)
│   │                                  #   3) Cost-Sensitive Threat Matrix
│   │
│   ├── ── Değerlendirme Scriptleri ───────────────────────────
│   ├── evaluate_router.py              # Router modelini değerlendirme
│   ├── evaluate_specialist.py          # Uzman modelleri değerlendirme
│   ├── build_eda_master_samples.py     # EDA için örnek veri seti oluşturma
│   ├── validate_router_mapping.py      # Router feature mapping doğrulama
│   │
│   ├── ── Test / Doğrulama Scriptleri ────────────────────────
│   ├── sanity_check_overfit.py         # Overfit sağlık kontrolü
│   ├── test_coerce_numeric.py          # Sayısal dönüşüm testi
│   ├── test_inf_sweep.py               # Sonsuz değer tarama testi
│   ├── test_latency_metrics.py         # Gecikme metrikleri testi
│   ├── test_sample_export.py           # Örnek dışa aktarma testi
│   ├── test_operational_modules.py     # ★ YENİ — 3 operasyonel modülün birim testi
│   └── test_path_adaptation.py         # ★ YENİ — Kaggle/Local path adaptasyon testi
│
├── notebooks/                          # Jupyter / EDA Notebook'ları
│   └── eda/                            # Keşifsel Veri Analizi
│       ├── eda_common.py               # Ortak EDA yardımcı fonksiyonları
│       ├── 01_eda_cicids2018_test.ipynb # CICIDS-2018 hızlı EDA (test)
│       ├── 02_eda_ciciot2023_test.ipynb # CICIOT-2023 hızlı EDA (test)
│       ├── 03_eda_cicids2018.ipynb     # CICIDS-2018 tam EDA
│       └── 04_eda_ciciot2023.ipynb     # CICIOT-2023 tam EDA
│
├── data/                               # Veriler (büyük dosyalar .gitignore'da)
│   ├── raw/                            # Ham, işlenmemiş veriler
│   │   ├── cicids2018/                 # CICIDS-2018 Cloud trafik verisi
│   │   └── ciciot2023/                 # CICIOT-2023 IoT trafik verisi
│   │
│   ├── processed/                      # İşlenmiş / temizlenmiş veriler
│   │   ├── cloud/                      # Cloud işlenmiş veri
│   │   └── iot/                        # IoT işlenmiş veri
│   │
│   ├── splits/                         # Train / Validation / Test bölümleri
│   │   ├── cloud/                      # Cloud veri bölümleri
│   │   └── iot/                        # IoT veri bölümleri
│   │
│   └── eda_samples/                    # EDA için küçük örnek setler
│       ├── cloud_master.parquet        # Cloud EDA master örnek
│       ├── cloud_master_meta.json      # Cloud örnek meta bilgisi
│       ├── iot_master.parquet          # IoT EDA master örnek
│       └── iot_master_meta.json        # IoT örnek meta bilgisi (sınıf dağılımı)
│
├── artifacts/                          # Eğitilmiş modeller ve değerlendirme çıktıları
│   ├── models/                         # Kaydedilmiş model dosyaları
│   │   ├── sanity_check_model.pt       # Sanity check PyTorch modeli
│   │   │
│   │   ├── anomaly_detector/           # Anomali dedektörü checkpoint'leri
│   │   │
│   │   ├── cloud_specialist/           # Cloud uzman modeli (Keras)
│   │   │   ├── model.keras             # Son eğitilmiş model
│   │   │   ├── best_epoch_00X.keras    # En iyi epoch checkpoint'leri
│   │   │   └── preprocessor.joblib    # Eğitilmiş preprocessor
│   │   │
│   │   ├── iot_specialist/             # IoT uzman modeli (Keras)
│   │   │   ├── model.keras             # Son eğitilmiş model
│   │   │   ├── best_epoch_00X.keras    # En iyi epoch checkpoint'leri
│   │   │   └── preprocessor.joblib    # Eğitilmiş preprocessor
│   │   │
│   │   └── router/                     # Trafik yönlendirici modeli
│   │       └── router.pkl              # Scikit-learn router modeli
│   │
│   ├── evaluation/                     # Değerlendirme sonuçları
│   │   ├── cloud_metrics.json          # Cloud model metrikleri (F1, AUC vb.)
│   │   ├── cloud_confusion_matrix.json # Cloud karmaşıklık matrisi
│   │   ├── cloud_predictions.csv       # Cloud tahmin çıktıları
│   │   ├── iot_metrics.json            # IoT model metrikleri
│   │   ├── iot_confusion_matrix.json   # IoT karmaşıklık matrisi
│   │   ├── iot_predictions.csv         # IoT tahmin çıktıları
│   │   ├── router_metrics.json         # Router doğruluk metrikleri
│   │   ├── kaggle_full_summary.json    # ★ YENİ — Kaggle pipeline tam özet raporu
│   │   │                              #   (metrikler + latency + 3 operasyonel modül)
│   │   │
│   │   ├── cloud/                      # Cloud görsel çıktılar (Faz 5)
│   │   │   ├── best_cloud_model.pt
│   │   │   ├── cloud_training_curves.png
│   │   │   ├── cloud_confusion_matrix.png
│   │   │   ├── cloud_roc_pr_curves.png
│   │   │   ├── cloud_attention_heatmap.png
│   │   │   ├── cloud_confidence_calibration.png
│   │   │   ├── cloud_tsne.png
│   │   │   └── cloud_phase5_summary.json
│   │   │
│   │   └── iot/                        # IoT görsel çıktılar (Faz 5)
│   │       ├── best_iot_model.pt
│   │       ├── iot_training_curves.png
│   │       ├── iot_confusion_matrix.png
│   │       ├── iot_roc_pr_curves.png
│   │       ├── iot_attention_heatmap.png
│   │       ├── iot_confidence_calibration.png
│   │       ├── iot_tsne.png
│   │       └── iot_phase5_summary.json
│   │
│   └── samples/                        # İşlenmiş örnek çıktılar (dashboard için)
│       ├── processed_sample_cloud.csv  # Cloud örnek inference verisi
│       └── processed_sample_iot.csv    # IoT örnek inference verisi
│
├── tests/                              # Birim ve entegrasyon testleri
│   ├── __init__.py
│   ├── test_architectures.py           # Model mimari testleri
│   ├── test_loader.py                  # Veri yükleyici testleri
│   ├── test_preprocessing.py           # Ön işleme pipeline testleri
│   │
│   ├── test_data/                      # Veri modülü testleri
│   │   ├── test_router_features.py
│   │   └── test_router_mapping_validate.py
│   │
│   ├── test_evaluation/                # Değerlendirme modülü testleri
│   ├── test_models/                    # Model modülü testleri
│   │
│   └── test_training/                  # Eğitim modülü testleri
│       └── test_splits.py
│
├── docs/                               # Proje dokümantasyonu
│   ├── DEVELOPMENT_ROADMAP.md          # Geliştirme yol haritası (Faz 1-6)
│   └── PROJECT_STATUS_AND_NEXT_STEPS_DETAILED.md
│
└── logs/                               # Çalışma zamanı logları
    └── app.log                         # Uygulama log dosyası
```

---

## 🆕 Son Güncellemeler

| Dosya | Durum | Açıklama |
|-------|-------|---------|
| `scripts/run_kaggle_full.py` | **YENİ** | Kaggle GPU ortamı için tam eğitim + değerlendirme pipeline orchestrator'ı. 3 operasyonel analiz modülü içerir. |
| `scripts/test_operational_modules.py` | **YENİ** | `run_kaggle_full.py` içindeki 3 operasyonel modülün bağımsız birim testi. |
| `scripts/test_path_adaptation.py` | **YENİ** | `src/utils/config.py` path yönetiminin Kaggle/Local geçişini doğrulayan entegrasyon testi. |
| `src/utils/config.py` | **GÜNCELLENDİ** | Yeni public API: `is_kaggle()`, `get_project_root()`, `get_dataset_path()`, `get_artifacts_root()` — ortam-bilinçli path çözümleme. |
| `configs/datasets.yaml` | **GÜNCELLENDİ** | `kaggle_raw_path` ve `environment.kaggle/local.artifacts_root` anahtarları eklendi. |
| `artifacts/evaluation/kaggle_full_summary.json` | **YENİ** | Kaggle pipeline çalıştırmasının tam JSON özeti (metrikler + latency + operasyonel modül sonuçları). |

---

## 🗂️ Modüller ve Sorumluluklar

| Klasör | Sorumluluk |
|--------|-----------|
| `src/data/` | Ham verinin yüklenmesi, temizlenmesi, feature engineering |
| `src/models/` | Sinir ağı mimarileri (Cloud, IoT, Anomaly/VAE, Ensemble) |
| `src/training/` | Eğitim döngüsü, early stopping, checkpoint kaydetme |
| `src/evaluation/` | Metrik hesaplama, XAI (SHAP/Attention), görselleştirme |
| `src/utils/` | Config yükleme, loglama, seed yönetimi, **Kaggle/Local path adaptasyonu** |
| `dashboard/` | Streamlit tabanlı gerçek zamanlı izleme arayüzü |
| `scripts/` | Eğitim, değerlendirme, Kaggle pipeline ve test scriptleri |
| `configs/` | YAML ile merkezi konfigürasyon (local + Kaggle ortam ayarları) |
| `notebooks/eda/` | Keşifsel veri analizi Jupyter notebook'ları |
| `data/` | Ham, işlenmiş ve bölünmüş veri setleri |
| `artifacts/` | Eğitilmiş modeller, metrikler ve görsel çıktılar |
| `tests/` | Pytest birim ve entegrasyon test paketi |
| `docs/` | Yol haritası ve proje durum belgeleri |

---

## 🧠 Sistem Mimarisi

```
Ağ Trafiği Gelir
       │
       ▼
  [ Router ]  ──── Trafik tipini belirler (Cloud mu? IoT mu?)
       │
       ├──► Cloud ──► [ Cloud Specialist ]  ──► Saldırı Tipi / Normal
       │
       └──► IoT   ──► [ IoT Specialist   ]  ──► Saldırı Tipi / Normal
                               │
                      Benign mi? Emin değilsek?
                               │
                               ▼
                    [ VAE Anomaly Detector ]  ──► Override → Saldırı / Normal
                               │
                               ▼
                    [ Cost-Sensitive Threat Matrix ]
                    (SOC operasyonel maliyet analizi)
```

| Bileşen | Teknoloji | Dataset |
|---------|-----------|---------|
| Router | Scikit-learn (RandomForest/XGBoost) | CICIDS-2018 + CICIOT-2023 |
| Cloud Specialist | PyTorch + Keras (Deep NN + Attention) | CICIDS-2018 |
| IoT Specialist | PyTorch + Keras (Deep NN + Attention) | CICIOT-2023 |
| Anomaly Detector (VAE) | Autoencoder / VAE (Keras) | Her iki dataset |
| Ensemble | Router + Specialist + VAE Override | — |
| Dashboard | Streamlit | — |
| Deney Takibi | MLflow | — |
| Kaggle Eğitimi | `run_kaggle_full.py` (GPU) | Her iki dataset |

---

## ⚙️ Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Veri setlerini data/raw/ altına koy
#    - CICIDS-2018  →  data/raw/cicids2018/
#    - CICIOT-2023  →  data/raw/ciciot2023/

# 3. Modelleri sırasıyla eğit (local)
python scripts/train_router.py
python scripts/train_cloud.py
python scripts/train_iot.py
python scripts/train_anomaly.py

# 4. Kaggle GPU ortamında tam pipeline (tercih edilen)
python scripts/run_kaggle_full.py --domain both --epochs 50 --batch-size 512

# 5. Operasyonel modülleri test et
python scripts/test_operational_modules.py
python scripts/test_path_adaptation.py

# 6. Dashboard'u başlat
streamlit run dashboard/app.py
```

---

> 📌 **Not:** `data/raw/`, `data/processed/`, `data/splits/` ve büyük model dosyaları `.gitignore` kapsamındadır. Bu klasörlerin içeriği Git'e gönderilmez; sadece yapı (`.gitkeep`) dosyaları tutulur.
