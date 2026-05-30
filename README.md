# AI-Driven Network Attack Detection

Akış tabanlı ağ trafiğinde saldırı tespiti için **ensemble** mimarisi: trafik önce bir **router** (IoT vs Cloud) ile yönlendirilir, ardından domain **uzman modelleri** (CNN + LSTM + attention) ve **VAE + Isolation Forest** tabanlı anomali skoru birleştirilir. Veri setleri **CIC-IoT2023** (IoT) ve **CIC-IDS2018** (Cloud) ile uyumludur.

## Gereksinimler

- Python 3.11+
- Bağımlılıklar: `requirements.txt`

## Kurulum

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Proje kökünden Python paket yolu olarak çalıştırın (`python scripts/...` veya `python -m pytest`).

## Veri dizinleri

Ham CSV dosyaları repoda yer almaz (`.gitignore`). Yapı `configs/datasets.yaml` ile tanımlıdır:

| Domain | Klasör (proje köküne göre) |
|--------|----------------------------|
| IoT (CIC-IoT2023) | `data/raw/ciciot2023/` |
| Cloud (CIC-IDS2018) | `data/raw/cicids2018/` |

IoT tarafında etiket çoğu zaman dosya adından türetilir (`label_source: filename`). Cloud tarafında etiket CSV içindeki `Label` sütunundadır.

## Router öznitelik eşlemesi

IoT ve Cloud sütun adları farklı; `configs/router.yaml` içindeki `feature_mapping` bu iki dünyayı ortak bir özellik uzayına çevirir.

Eğitim veya üretim öncesi başlıkları doğrulamak için:

```bash
python scripts/validate_router_mapping.py
```

Eksik sütun varsa çıkış kodu ile başarısız olması için:

```bash
python scripts/validate_router_mapping.py --strict
```

Gerekirse `configs/router.yaml` içindeki `iot` / `cloud` alanlarını gerçek CSV başlıklarına göre güncelleyin.

## Eğitim sırası (önerilen)

Tüm komutlar **proje kökünden** çalıştırılır. Büyük veri için GPU ve yeterli RAM önerilir.

1. **Router** (IoT vs Cloud ikili sınıflandırıcı):

   ```bash
   python scripts/train_router.py
   ```

   Hızlı deneme: `python scripts/train_router.py --nrows 50000`

2. **IoT uzmanı**:

   ```bash
   python scripts/train_iot.py
   ```

3. **Cloud uzmanı**:

   ```bash
   python scripts/train_cloud.py
   ```

4. **Anomali modelleri** (IoT ve Cloud özellik uzayları farklı olduğu için **iki ayrı** eğitim):

   ```bash
   python scripts/train_anomaly.py --domain iot
   python scripts/train_anomaly.py --domain cloud
   ```

Çıktılar `artifacts/models/` altında toplanır (ör. `router/router.pkl`, `iot_specialist/model.keras`, `anomaly_detector_iot/`, `anomaly_detector_cloud/`). `configs/ensemble.yaml` içindeki `anomaly_models` bu klasör adlarıyla uyumludur.

### Değerlendirme (test split)

Eğitilmiş uzman + `preprocessor.joblib` ile aynı split mantığında test kümesinde metrik üretmek için:

```bash
python scripts/evaluate_specialist.py --domain iot
python scripts/evaluate_specialist.py --domain cloud
```

Çıktı: `artifacts/evaluation/{iot|cloud}_metrics.json`, `*_confusion_matrix.json`, `*_predictions.csv` (Streamlit Metrikler sayfasına `predictions.csv` olarak yüklenebilir). `--nrows` ve `--seed` eğitimle aynı tutulursa test kümesi eğitimle tutarlı olur.

### Router değerlendirme

```bash
python scripts/evaluate_router.py
# Hızlı deneme (`--nrows` her .csv dosyası için ayrıdır; çok dosya = çok satır):
python scripts/evaluate_router.py --nrows 5000
# Eski davranış: dosya başına INFO logları için --verbose
```

Çıktı: `artifacts/evaluation/router_metrics.json` (IoT/Cloud test parçalarında domain doğruluğu, gecikme). **Not:** `train_router` tüm veriyi kullanıyorsa test satırları eğitimde görülmüş olabilir; dosyadaki `note` alanına bakın.

### Monitoring dashboard

`evaluate_*` scriptleri çalıştıktan sonra Streamlit **Monitoring** sayfası `artifacts/evaluation/*.json` özetlerini gösterir.

## Dashboard (4 panel)

```bash
streamlit run dashboard/app.py
```

| Sayfa | İçerik |
|--------|--------|
| Ana | Artifact durumu |
| Monitoring | İleride MLflow / metrik kancaları |
| Alerts | `ensemble.yaml` uyarı eşikleri |
| **XAI** | SHAP ile top-k özellik önemi (IoT/Cloud uzmanı; CSV veya sentetik demo) |
| **Metrikler** | `y_true` / `y_pred` (ve isteğe bağlı `proba_*`) CSV yükleme, confusion matrix, rapor |

Uzman modeli eğitildiğinde `preprocessor.joblib` aynı klasöre kaydedilir (`train_iot.py` / `train_cloud.py`); XAI sayfasında ham CSV dönüşümü için gereklidir. Modeller yokken XAI sayfası sentetik veri demosu gösterebilir.

Modeller eğitilmeden bileşen durumu “missing” görünebilir; bu beklenen bir durumdur.

## Testler

```bash
python -m pytest tests/ -v
```

Varsayılan `pyproject.toml` ayarı coverage içerir; `pytest-cov` yüklü değilse:

```bash
python -m pytest tests/ -v --no-cov
```

## Yapılandırma özeti

| Dosya | İçerik |
|--------|--------|
| `configs/datasets.yaml` | Veri yolları, hedef sütunu, sınıf listeleri |
| `configs/router.yaml` | Router algoritması ve `feature_mapping` |
| `configs/iot_model.yaml` / `configs/cloud_model.yaml` | Uzman mimarileri |
| `configs/anomaly_detector.yaml` | VAE ve Isolation Forest |
| `configs/ensemble.yaml` | Anomali entegrasyonu, uyarı eşikleri, çift anomali klasör adları |
| `configs/training.yaml` | Epoch, batch, MLflow, sınıf ağırlığı |

## Lisans ve atıf

Proje kapsamında kullanılan CIC veri setleri için ilgili kurumların kullanım koşullarına uyun.
