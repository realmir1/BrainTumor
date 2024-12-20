# 🧠 Keras ile Beyin Tümörü Tespiti 🧠  


## İçindekiler  
- [🔍 Proje Genel Bakışı](#-proje-genel-bakışı)  
- [🎯 Amaçlar](#-amaçlar)  
- [🧰 Kullanılan Teknolojiler](#-kullanılan-teknolojiler)  
  - [Keras](#keras)  
  - [Diğer Kütüphaneler](#diğer-kütüphaneler)  
- [📁 Proje Yapısı](#-proje-yapısı)  
- [🛠️ Kurulum ve Çalıştırma](#️-kurulum-ve-çalıştırma)  
- [💻 Kullanım](#-kullanım)  
  - [Veri Hazırlama](#veri-hazırlama)  
  - [Model Mimarisi](#model-mimarisi)  
  - [Modeli Eğitme](#modeli-eğitme)  
  - [Modeli Değerlendirme](#modeli-değerlendirme)  
  - [Tahminler Yapma](#tahminler-yapma)  
- [📈 Sonuçlar](#-sonuçlar)  
- [📚 Kaynaklar](#-kaynaklar)  
- [📝 Katkıda Bulunma](#-katkıda-bulunma)  
- [📜 Lisans](#-lisans)  
- [👤 Yazar](#-yazar)  

---

## 🔍 Proje Genel Bakışı  
Bu proje, **Keras** kullanılarak geliştirilen bir **Evrişimsel Sinir Ağı (CNN)** modeliyle **beyin MRI görüntülerindeki tümörleri tespit etmeyi** amaçlamaktadır. Geliştirilen model, MRI taramalarını **"Tümör Yok"** ve **"Tümör Var"** olarak sınıflandırır. Modelin etkinliği ve doğruluğu, sağlık profesyonellerine erken teşhis için yardımcı olabilir.  

---

## 🎯 Amaçlar  
- **Veri İşleme**: MRI görüntülerinin yüklenmesi ve ön işlenmesi.  
- **Model Geliştirme**: Keras kullanarak bir CNN modeli oluşturma.  
- **Model Değerlendirme**: Modelin doğruluğunu görselleştirerek performansını analiz etme.  
- **Tahmin Gösterimi**: Modelin yaptığı tahminleri örneklerle sunma.  

---

## 🧰 Kullanılan Teknolojiler  

### Keras  
**Keras**, Python dilinde çalışan, hızlı prototipleme için ideal olan, derin öğrenme modelleri geliştirmeye yönelik yüksek seviyeli bir kütüphanedir. TensorFlow gibi arka uçlarla çalışarak performansı artırır.  

**Keras'ın Özellikleri:**  
- **Kullanıcı Dostu API**: Karmaşık ağları kolayca oluşturmanızı sağlar.  
- **Modüler Yapı**: Katmanlar, aktivasyon fonksiyonları ve optimizasyonlar modüler yapıdadır.  
- **Esnek ve Genişletilebilir**: Geliştiricilere özel bileşenler ekleme imkanı sunar.  

### Diğer Kütüphaneler  
- **Python 3.x**: Ana programlama dili.  
- **OpenCV**: Görüntü işleme kütüphanesi.  
- **NumPy**: Matematiksel işlemler için.  
- **Matplotlib**: Grafikler ve görselleştirme için.  
- **Scikit-learn**: Veri bölme ve model değerlendirme araçları.  

---

## 📁 Proje Yapısı  

```
beyin-tumor-tespiti/  
├── data/  
│   ├── no/  
│   └── yes/  
├── images/  
│   ├── brain_mri.jpg  
│   └── sonuç_grafiği.png  
├── notebooks/  
│   └── tumor_detection.ipynb  
├── requirements.txt  
├── README.md  
└── LICENSE  
```

- **data/**: MRI görüntülerinin bulunduğu klasör.  
- **images/**: Sonuç grafiklerinin ve görsellerin tutulduğu klasör.  
- **notebooks/**: Tüm kodların bulunduğu Jupyter Notebook dosyası.  
- **requirements.txt**: Proje için gerekli kütüphaneler listesi.  
- **README.md**: Proje dokümantasyonu.  
- **LICENSE**: Lisans bilgileri.  

---

## 🛠️ Kurulum ve Çalıştırma  

### Gereksinimler  
- **Python 3.6+**  
- **pip** paket yöneticisi  

### Kurulum Adımları  

1. **Proje Deposunu Klonlayın**  
   ```bash
   git clone https://github.com/yourusername/brain-mri-tumor-detection.git  
   cd brain-mri-tumor-detection  
   ```  

2. **Sanallaştırılmış Ortam Oluşturun**  
   ```bash
   python3 -m venv venv  
   source venv/bin/activate  # Windows için: venv\Scripts\activate  
   ```  

3. **Bağımlılıkları Kurun**  
   ```bash
   pip install -r requirements.txt  
   ```  

4. **Veri Kümesini İndirin**  
   - Veriyi [Kaggle'dan](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) indirin ve `data/` klasörüne yerleştirin.  

---

## 💻 Kullanım  

### Veri Hazırlama  
Veriler OpenCV kullanılarak gri tonlamaya çevrilir, boyutlandırılır ve normalleştirilir.  

### Model Mimarisi  
Aşağıdaki CNN mimarisi kullanılmıştır:  
- **Conv2D**: 32 ve 64 filtre ile iki katman.  
- **MaxPooling2D**: Özellik haritalarını küçültme.  
- **Flatten**: Düzleştirme katmanı.  
- **Dense**: Tam bağlantılı katmanlar.  
- **Dropout**: %50 oranında rastgele nöron bırakma.  

---

### Modeli Eğitme  
```python
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
```

---

### Modeli Değerlendirme  
Test doğruluğu ölçülüp ekrana yazdırılır.  

---

## 📈 Sonuçlar  

### Eğitim ve Doğrulama Doğruluğu  
Eğitim sırasında modelin doğruluğu:  

![Model Accuracy](https://github.com/yourusername/brain-mri-tumor-detection/blob/main/images/sonuç_grafiği.png)  

- **Eğitim Doğruluğu**: %92  
- **Test Doğruluğu**: %90  

---

## 📚 Kaynaklar  
- [Keras Dokümantasyonu](https://keras.io)  
- [TensorFlow Kılavuzu](https://www.tensorflow.org/guide/keras)  
- [Kaggle Beyin MRI Verisi](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  

---



---

## 📜 Lisans  
Bu proje **MIT Lisansı** altında lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına göz atın.  

---


---

> **Not**: Bu proje eğitim amaçlıdır. Kesin tıbbi teşhis için sağlık uzmanlarına başvurunuz.
