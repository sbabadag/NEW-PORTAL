# Portal Frame Analysis Tool

Bu araç, portal çerçeve sistemlerinin analizi için Eurocode benzeri kar şekillendirme ve basit rüzgar basıncı ile N-V-M diyagramları oluşturur.

## Özellikler

- **Geometri**: Portal çerçeve geometrisi (açıklık, kolon yükseklikleri, mahya)
- **Kesitler**: Kolon ve kiriş kesit özellikleri (A, I) - IPE/HEA etiketleri
- **Yükler**:
  - Ölü yük G [kN/m²] (düşey)
  - Kar yükü (EN 1991-1-3 tarzı μ(α) ile): s = μ × Ce × Ct × s_k
  - Rüzgar yükü (çatı-normal basınç) [kN/m²]
- **Kombinasyonlar**:
  - ULS (G + S): 1.35×G + 1.50×S
  - ULS (G + W): 1.35×G + 1.50×W
  - SLS (G + S): 1.00×G + 1.00×S
  - SLS (G + W): 1.00×G + 1.00×W

## Kurulum

### Otomatik Kurulum
1. `setup.bat` dosyasını çalıştırın
2. Kurulum tamamlandıktan sonra `run_portal.bat` ile programı başlatın

### Manuel Kurulum
```bash
# Sanal ortam oluşturun
python -m venv .venv

# Sanal ortamı etkinleştirin
.venv\Scripts\activate

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

## Kullanım

### 🎯 Önerilen: Modern UI (CustomTkinter)
```bash
# run_portal.bat çalıştırın ve seçenek 1'i seçin
run_portal.bat

# Veya doğrudan:
.venv\Scripts\python.exe portal_modern_ui.py
```
- ✅ Modern dark theme arayüz
- ✅ Gerçek zamanlı veri giriş kutuları ve etiketler
- ✅ Dropdown menüler ve onay kutuları
- ✅ Tüm grafikler tek sayfada bölümlenmiş pencereler
- ✅ Pik değerlerde kuvvet değerleri etiketli gösterim
- ✅ Entegre sonuç görüntüleme

### 📓 Alternatif: İnteraktif Jupyter Notebook
```bash
# run_portal.bat çalıştırın ve seçenek 3'ü seçin
run_portal.bat

# Veya doğrudan:
.venv\Scripts\jupyter notebook portal_interactive.ipynb
```
- ✅ Gerçek zamanlı veri giriş kutuları
- ✅ Dropdown menüler ve onay kutuları
- ✅ Tek tıkla analiz
- ✅ Entegre sonuç görüntüleme

### 📟 Konsol Versiyonu
```bash
# run_portal.bat çalıştırın ve seçenek 2'yi seçin
run_portal.bat

# Veya doğrudan:
.venv\Scripts\python.exe portal.py
```
- ✅ Terminal/komut satırında çalışır
- ✅ Adım adım parametre girişi
- ✅ Jupyter olmadan çalışır

### VS Code'dan Kullanım
- **Ctrl+Shift+P** → "Tasks: Run Task" → İstediğiniz görevi seçin:
  - "Run Portal Analysis (Modern UI)" - Modern CustomTkinter arayüzü
  - "Run Portal Analysis (Console)" - Konsol versiyonu
  - "Run Portal Analysis (Jupyter)" - İnteraktif versiyon
  - "Test Environment" - Ortam testi

## Üç Versiyon Karşılaştırması

| Özellik | Modern UI | Jupyter Notebook | Konsol Versiyonu |
|---------|-----------|------------------|-------------------|
| Veri Girişi | Edit kutuları + etiketler | İnteraktif kutular | Soru-cevap |
| Grafik Görünüm | Tek sayfa, 4 bölümlenmiş pencere | Ayrı figürler | Ayrı figürler |
| Kuvvet Değerleri | Pik noktalarda etiketli | Standart | Standart |
| Tema | Modern dark theme | Jupyter default | Terminal |
| Kullanım | Tek tık | Tek tık | Adım adım |
| Gereksinim | CustomTkinter | Jupyter gerekli | Sadece Python |
| Deneyim | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Modern UI Özellikleri

### 🎨 Arayüz
- **Dark Theme**: Modern görünüm
- **Sol Panel**: Tüm parametreler organize edilmiş bölümler halinde
- **Sağ Panel**: Tabullu sonuç görüntüleme

### 📊 Grafik Gösterimi
- **Geometri**: Çerçeve şekli, mesnetler ve eleman etiketleri
- **Eksenel Kuvvet (N)**: Pik değerlerde kN cinsinden etiketler
- **Kesme Kuvveti (V)**: Pik değerlerde kN cinsinden etiketler
- **Moment (M)**: Pik değerlerde kN⋅m cinsinden etiketler

### 🔧 Kontroller
- **Edit Kutuları**: Sayısal değer girişi
- **Dropdown Menüler**: Kesit seçimi (HEA, IPE)
- **Onay Kutuları**: Yük kombinasyonları ve seçenekler
- **Hesapla Butonu**: Tek tıkla analiz

## Gereksinimler

- Python 3.8+
- NumPy >= 1.24.0
- Matplotlib >= 3.6.0
- CustomTkinter >= 5.0.0 (modern UI için)
- Pillow >= 9.0.0 (modern UI için)
- IPywidgets >= 8.0.0 (interaktif arayüz için)
- IPython >= 8.0.0
- Jupyter >= 1.0.0 (notebook versiyonu için)

## Dosya Yapısı

```
PORTAL/
├── portal_modern_ui.py         # Modern UI (CustomTkinter) - Ana önerilen
├── portal.py                   # Konsol versiyonu
├── portal_interactive.ipynb    # Jupyter notebook versiyonu
├── requirements.txt            # Python bağımlılıkları
├── setup.bat                  # Otomatik kurulum
├── run_portal.bat             # Ana başlatıcı (tüm versiyonlar)
├── run_modern_ui.bat          # Doğrudan modern UI başlatıcı
├── test_environment.py        # Ortam test scripti
└── README.md                  # Bu dosya
```

## Hızlı Başlangıç

1. **İlk kurulum**: `setup.bat` çalıştırın
2. **Program başlatma**: `run_portal.bat` çalıştırın
3. **Versiyon seçimi**: 
   - **Seçenek 1**: Modern UI (önerilen)
   - Seçenek 2: Konsol versiyonu
   - Seçenek 3: Jupyter notebook

## Sorun Giderme

### "Modern UI açılmıyor"
- **Çözüm**: CustomTkinter yüklü olduğundan emin olun: `setup.bat` tekrar çalıştırın

### "Grafiklerde değerler görünmüyor"
- **Çözüm**: Modern UI versiyonunu kullanın, pik değerler otomatik etiketlenir

### "Python bulunamıyor"
- **Çözüm**: Python'un yolda (PATH) olduğundan emin olun

### "Paket yüklenememiyor"
- **Çözüm**: İnternet bağlantınızı kontrol edin, `setup.bat` tekrar çalıştırın

## Test

Ortamınızı test etmek için:
```bash
.venv\Scripts\python.exe test_environment.py
```
