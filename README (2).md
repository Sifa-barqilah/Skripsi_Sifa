# Segmentasi Pelanggan K-Means — Queen Buket Majalaya (Streamlit)

Versi Streamlit dari aplikasi segmentasi pelanggan, dengan tambahan:

- Grafik **Elbow Method (WCSS)** dan **Silhouette Score** berdampingan pada satu
  grafik (dua sumbu-Y), lengkap dengan garis bantu vertikal untuk titik siku
  optimal dan k dengan Silhouette Score terbaik.
- Interpretasi & rekomendasi strategi otomatis oleh model **AI Google (Gemini)**
  — pengguna memasukkan API Key sendiri, memilih model dari daftar yang dimuat
  secara dinamis dari akun Google AI mereka, lalu model tersebut menganalisis
  hasil clustering dan memberi rekomendasi strategi pemasaran.

## Cara Menjalankan

1. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

4. Browser akan terbuka otomatis di `http://localhost:8501`.

## Mendapatkan Google AI API Key

1. Buka https://aistudio.google.com/apikey
2. Login dengan akun Google, klik "Create API Key".
3. Salin key tersebut, tempel di kolom "Google AI API Key" pada tab
   **5️⃣ Hasil & Interpretasi AI**.
4. API Key hanya dipakai selama sesi berjalan di memori aplikasi (tidak
   ditulis ke file/disk oleh aplikasi ini).

## Alur Aplikasi

1. **Data** — gunakan data transaksi sintetis (~350–500 baris, meniru pola
   penjualan nyata lengkap dengan duplikat & nilai kosong), atau unggah CSV
   sendiri.
2. **Preprocessing** — pembersihan duplikat & nilai kosong, agregasi ke level
   pelanggan (frekuensi pembelian, jumlah transaksi, total nilai pembelian).
3. **Normalisasi** — Min-Max scaling ke rentang 0–1.
4. **Elbow & Clustering** — hitung WCSS & Silhouette Score untuk rentang k,
   lihat grafik gabungan dengan garis bantu k optimal, pilih k final, jalankan
   K-Means.
5. **Hasil & Interpretasi AI** — profil segmen, visualisasi scatter, tabel
   hasil (dapat diunduh CSV), dan analisis strategi oleh Gemini.

## Catatan Teknis

- K-Means & perhitungan Silhouette Score menggunakan `scikit-learn`
  (`random_state` tetap agar hasil dapat direproduksi).
- Klien AI menggunakan SDK resmi terbaru `google-genai`
  (`pip install google-genai`, `from google import genai`) — bukan paket lama
  `google-generativeai` yang sudah deprecated.
