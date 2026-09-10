# 🍈 Deteksi Penyakit Daun Melon

Aplikasi web berbasis **Streamlit** untuk mendeteksi penyakit pada daun melon menggunakan model **YOLO (You Only Look Once)**. Dibuat sebagai bagian dari penulisan ilmiah/tugas akhir.

🔗 **Live Demo:** [deteksi-penyakit-daun-melon.streamlit.app](https://deteksi-penyakit-daun-melon.streamlit.app/)

## ✨ Fitur

- 🔐 **Login & Registrasi** pengguna
- 📤 **Deteksi via Upload Gambar** — unggah foto daun melon dan dapatkan hasil deteksi penyakit beserta tingkat kepercayaan (confidence)
- 📷 **Deteksi via Webcam Real-time** — deteksi langsung menggunakan kamera perangkat (via `streamlit-webrtc`)
- 📊 **Riwayat Deteksi** — histori hasil deteksi tersimpan per pengguna dengan timestamp (zona waktu WIB)
- ℹ️ **Info Aplikasi** — halaman tentang aplikasi dan model yang digunakan

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Framework Web | [Streamlit](https://streamlit.io/) |
| Model Deteksi | YOLO (Ultralytics) |
| Real-time Video | streamlit-webrtc, OpenCV, PyAV |
| Database | SQLite |
| Manipulasi Data | Pandas, NumPy |
| Zona Waktu | Pytz |

## 📁 Struktur Project
├── app.py # Entry point aplikasi Streamlit
├── database.py # Fungsi database (login, register, riwayat deteksi)
├── model_load.py # Loader model YOLO (best.pt)
├── ui_functions.py # Kumpulan halaman/komponen UI
├── webcam_processor.py # Processor video real-time untuk deteksi via webcam
├── best.pt # Model YOLO hasil training
└── requirements.txt # Daftar dependency Python


## 🚀 Instalasi & Menjalankan Secara Lokal

1. **Clone repository**
```bash
   git clone https://github.com/<username>/deteksi_penyakit_daun_melon.git
   cd deteksi_penyakit_daun_melon
```

2. **Buat virtual environment (opsional tapi disarankan)**
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```

3. **Install dependency**
```bash
   pip install -r requirements.txt
```

4. **Jalankan aplikasi**
```bash
   streamlit run app.py
```

5. Buka browser ke `http://localhost:8501`

## 🩺 Kelas Penyakit yang Dideteksi

Penyakit daun melon yang dideteksi terbatas pada 3 kelas, yaitu:
- **Downy Mildew**
- **Cucumber Mosaic Virus (CMV)**
- **Daun Sehat**

## 📌 Catatan Deployment

Aplikasi ini di-deploy menggunakan **Streamlit Community Cloud**. Karena menggunakan `opencv-python-headless`, tidak diperlukan file `packages.txt` tambahan untuk dependency sistem (libGL).

## 👤 Kontributor

- Mariana Pangaribuan — Penulisan Ilmiah / Tugas Akhir

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik.