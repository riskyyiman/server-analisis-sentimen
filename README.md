# Server Analisis Sentimen (Sentiment Analysis Backend)

![Python](https://img.shields.io/badge/Python-100%25-blue?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

Backend server untuk melakukan analisis sentimen pada teks. Proyek ini dibangun menggunakan **Python** dan dirancang untuk memproses input teks serta mengklasifikasikannya (contoh: Positif, Negatif, Netral) menggunakan model Machine Learning/NLP.

## 📋 Daftar Isi
- [Tentang Proyek](#-tentang-proyek)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Prasyarat](#-prasyarat)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Dokumentasi API](#-dokumentasi-api)
- [Struktur Folder](#-struktur-folder)


## 📖 Tentang Proyek
Repository ini berfungsi sebagai API server yang menerima request berupa teks dari sisi klien (Frontend/Mobile App), memprosesnya melalui model analisis sentimen, dan mengembalikan hasil prediksi.

**Fitur Utama:**
* RESTful API untuk prediksi sentimen.
* Pemrosesan teks (Preprocessing).
* Respon JSON yang cepat dan ringan.


## 🛠 Teknologi yang Digunakan
* **Bahasa:** Python 3.x
* **Framework Web:** Flask / FastAPI (Sesuaikan dengan framework yang Anda gunakan)
* **Machine Learning:** Scikit-learn / NLTK / TensorFlow (Sesuaikan)
* **Environment:** Virtualenv / Conda


## ⚙️ Prasyarat
Sebelum memulai, pastikan Anda telah menginstal:
* [Python](https://www.python.org/downloads/) (versi 3.8 atau lebih baru)
* [Git](https://git-scm.com/)


## 🚀 Instalasi

Ikuti langkah-langkah berikut untuk menjalankan server di lokal:

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/riskyyiman/server-analisis-sentimen.git](https://github.com/riskyyiman/server-analisis-sentimen.git)
    cd server-analisis-sentimen
    ```

2.  **Buat Virtual Environment (Disarankan)**
    ```bash
    # Untuk Windows
    python -m venv venv
    venv\Scripts\activate

    # Untuk macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Pastikan Anda memiliki file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Cara Menjalankan

Setelah instalasi selesai, jalankan server dengan perintah:

```bash
# Contoh jika menggunakan Flask
python app.py

# ATAU jika menggunakan FastAPI/Uvicorn
uvicorn main:app --reload
