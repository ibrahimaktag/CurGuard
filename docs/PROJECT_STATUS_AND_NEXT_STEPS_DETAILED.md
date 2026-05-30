# Proje Durumu ve Sunum Hazırlık Notları (Detaylı)

Bu belge, `DEVELOPMENT_ROADMAP.md` dosyasının daha açıklamalı versiyonudur.

## 1) Bu belge ne, ne değil?

- Bu belge bir **ekip içi hazırlık notudur**.
- Bu belge bir **uygulama kılavuzu değildir**.
- Bu belge adım adım "şu komutu çalıştır" dokümanı değildir.
- Bu belge, projeyi anlatırken ekip üyelerinin aynı dili konuşmasını sağlamak içindir.

### Uygulama kılavuzu neden değil?

- Kod tarafında komutlar zaten `README.md` içinde var.
- Bu dosyanın ana amacı teknik kararların mantığını ve sunum dilini netleştirmek.
- Sunum öncesi "neleri kesin bilmeliyiz?" sorusuna cevap vermek.

---

## 2) Projenin şu anki durumu (bitmiş kısım)

### 2.1 Projenin yaptığı iş (tek cümle)

- Gelen ağ akış kayıtlarını (CSV) analiz edip trafiği IoT/Cloud olarak yönlendirir, ilgili uzman modelle saldırı sınıfını tahmin eder, anomali skoru üretir ve sonuçları dashboard'da gösterir.

### 2.2 Teknik olarak şu an çalışan ana parçalar

- Veri yükleme ve temizleme (`loader`, `preprocessor` katmanı).
- IoT ve Cloud için ayrı uzman sınıflandırıcılar.
- Router ile otomatik domain seçimi (IoT vs Cloud).
- Ensemble katmanı ile uzman + anomali sinyali birleştirme.
- VAE + Isolation Forest hibrit anomali skoru.
- SHAP tabanlı açıklama (XAI paneli).
- Streamlit dashboard (monitoring, alerts, xai, metrics).
- Eğitim ve değerlendirme scriptleri.
- Router mapping doğrulama scripti.

### 2.3 Ekip olarak özellikle bilmeniz gereken sınırlar

- Sistem canlı paket yakalayıp anlık işlemiyor; temel akış CSV üstünden.
- Yani şu an "offline/nearline" çalışma mantığı var.
- Başka kurum ağına tak-çalıştır ürün seviyesinde değil; araştırma/bitirme projesi seviyesinde.
- Müşteri ağı farklıysa model performansı düşebilir; gerekirse yeniden eğitim/fine-tune gerekir.

---

## 3) Projenin sonraki geliştirme aşaması (planlanan)

Bu kısım, bir üst sürümde eklenecek geliştirmeleri açıklar.

### 3.1 Faz A - Dağıtım iyileştirmesi

- Docker ile kurulum standartlaştırma.
- `.dockerignore` ile gereksiz dosyaları image dışında tutma.
- İsteğe bağlı `docker-compose` ile tek komut çalıştırma.
- Beklenen fayda: ortam farkı kaynaklı kurulum sorunlarını azaltma.

### 3.2 Faz B - Inference API

- FastAPI ile health ve predict endpointleri.
- Dış sistemlerden programatik tahmin çağrısı.
- Temel güvenlik: dosya boyutu ve satır limiti.
- Beklenen fayda: sistemi dashboard dışı uygulamalara entegre edebilme.

### 3.3 Faz C - PCAP köprüsü (POC)

- `tshark` benzeri araçla ham trafik -> özet özellik üretimi.
- Üretilen çıktıyı mevcut pipeline ile tahmine sokma.
- Beklenen fayda: canlıya bir adım yaklaşan demo.
- Not: Bu, tam üretim IDS değildir; POC seviyesidir.

### 3.4 Faz D - İnce ayar/polish

- Paketleme ve import düzeni sadeleştirme.
- İsteğe bağlı CI (pytest/lint).
- Router değerlendirmesinde train/test ayrımını daha net hale getirme.

---

## 4) Sunum öncesi ekip olarak mutlaka öğrenmemiz gerekenler

Aşağıdaki başlıklar derin öğrenme teorisi ders notu değil; bu projeyi savunmak için gerekli minimum set.

### 4.1 Problem tanımı

- Neyi çözüyoruz? (IoT + Cloud ağ saldırı tespiti)
- Neden tek model değil, neden uzmanlar? (domain farkı)
- Neden payload yerine flow/header özellikleri? (gizlilik + uygulanabilirlik)

### 4.2 Veri ve etiketleme mantığı

- IoT verisinde etiketin dosya adından türetildiği durum.
- Cloud verisinde `Label` sütununun kullanımı.
- Dengesiz sınıf problemi ve class weights yaklaşımı.

### 4.3 Model mimarisi mantığı

- Router ne yapar, hangi durumda `both` döner?
- Uzman modeller neyi tahmin eder?
- Anomali modeli neden ayrı bir sinyal olarak var?
- VAE + Isolation Forest hibriti neden seçildi?

### 4.4 Değerlendirme ve metrikler

- Accuracy tek başına neden yetmez?
- Weighted F1, per-class F1 neden önemli?
- ROC-AUC ve PR-AUC ne zaman anlamlı olur?
- Confusion matrix nasıl yorumlanır?

### 4.5 Operasyonel gerçeklik

- Projenin canlı sistem değil, araştırma prototipi olduğu net söylenmeli.
- "Kendi ağında kullanmak isteyen" kişi için eğitim/fine-tune gereksinimi açıklanmalı.
- Mevcut sınırlamalar ve gelecek çalışma maddeleri dürüstçe belirtilmeli.

---

## 5) Deep Learning hocasına götürürken sorulacak sorular

Bu bölüm direkt toplantıda kullanabileceğiniz soru havuzu.

### 5.1 Mimari doğrulama soruları

- IoT/Cloud için iki uzman + router yaklaşımı akademik olarak yeterince savunulabilir mi?
- Ensemble'da ağırlıklandırma stratejisini daha doğru yapmak için ne önerirsiniz?
- `both` fallback için daha iyi karar mekanizması önerir misiniz?

### 5.2 Model tasarımı soruları

- CNN+LSTM+MHA seçimi bu veri yapısı için uygun mu?
- Ablation için en az hangi karşılaştırmayı yapmalıyız ki rapor güçlü olsun?
- Router için LightGBM yeterli mi, yoksa başka bir baseline da eklemeli miyiz?

### 5.3 Anomali tarafı soruları

- VAE+IF hibrit skoru için ağırlık seçimini nasıl daha bilimsel yapabiliriz?
- Domain başına ayrı anomali modeli kararımız doğru mu?
- Anomali threshold kalibrasyonu için hangi yöntemi önerirsiniz?

### 5.4 Değerlendirme ve rapor soruları

- Jüri önünde hangi metrikler mutlaka ana tabloda olmalı?
- Hangi grafikler (ROC, PR, confusion matrix, SHAP) sunumda en etkili olur?
- Veri seti zamansallığı (dataset age) eleştirisini raporda nasıl çerçevelemeliyiz?

### 5.5 Ürünleştirme/gelecek çalışma soruları

- Bitirme kapsamını aşmadan hangi ek geliştirme en yüksek etkiyi sağlar? (Docker/API/PCAP)
- Canlı trafik için POC seviyesinde en gerçekçi yol sizce hangisi?
- Gelecek çalışma bölümünü nasıl yazarsak akademik olarak güçlü görünür?

---

## 6) Hoca bize muhtemelen hangi soruları sorar?

### 6.1 Temel mimari soruları

- Neden tek model yerine bu mimari?
- Router hata yaparsa sistem nasıl davranıyor?
- Neden bu özellikleri seçtiniz?

### 6.2 Veri ve deney tasarımı soruları

- Train/val/test ayrımı nasıl yapıldı?
- Veri sızıntısı riskini nasıl kontrol ettiniz?
- Sınıf dengesizliğini nasıl yönettiniz?

### 6.3 Sonuç soruları

- En kritik metrikleriniz neler?
- Hangi sınıflarda zorlanıyorsunuz?
- Sistem neden bazı saldırılarda iyi, bazılarında zayıf?

### 6.4 Operasyon soruları

- Bu sistem gerçek ağda nasıl konumlanır?
- Şu anki hali canlı mı, offline mı?
- Müşteri bunu alırsa yeniden eğitim gerekir mi?

### 6.5 XAI soruları

- SHAP'i neden eklediniz?
- Analiste pratikte ne fayda sağlıyor?
- SHAP çıktısını yanlış yorumlama riski var mı?

---

## 7) Toplantıya gitmeden önce ekip içi kontrol listesi

### 7.1 Her ekip üyesi bilsin

- Projenin tek cümlelik tanımı.
- Veri kaynakları ve etiketleme farkı.
- Router + specialist + anomaly akışı.
- Ana metriklerin ne anlama geldiği.
- Projenin canlı sistem olmadığını ve nedenini.

### 7.2 Yanımızda hazır olsun

- Güncel mimari diyagram (tek sayfa).
- 1-2 sonuç tablosu (mümkünse specialist + router).
- Confusion matrix örneği.
- SHAP top-feature ekran görüntüsü.
- "Sınırlamalar ve gelecek çalışma" kısa listesi.

### 7.3 Cevaplarımızda dikkat

- Bilmediğimiz yerde tahmin değil, "bunu test etmedik" diyebilmek.
- Sonuçları abartmamak (özellikle %100 router sonucu için veri sızıntısı notu).
- Yapılan iş ile planlanan işi net ayırmak.

---

## 8) Kısa kapanış

- Mevcut repo, bitirme projesi için güçlü bir teknik omurga sunuyor.
- Bir üst seviyeye çıkmak için en rasyonel adımlar: **Docker + API**.
- Toplantılarda odak: "Ne yaptık? Neden doğru? Nerede sınır var? Sonra ne yapacağız?"

Bu belge, ekip içi hizalanma içindir; kod geliştirme ilerledikçe güncellenmelidir.
