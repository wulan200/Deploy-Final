# Clustering Algorithm Comparison and Seasonal Validation for Daily Weather Patterns in Bima Regency

Aplikasi web **Streamlit** untuk mendemonstrasikan penelitian yang telah dipublikasikan:
perbandingan empat algoritma klasterisasi (K-Means, Gaussian Mixture Model, Self-Organizing Map,
Hierarchical/Ward) pada data cuaca harian NASA POWER Kabupaten Bima, NTB (2014–2025), beserta
validasi eksternal terhadap musim.

> Aplikasi ini **tidak mengubah metodologi penelitian**. Seluruh preprocessing, algoritma, parameter,
> metrik, dan validasi musiman dipindahkan dari notebook penelitian tanpa modifikasi.

## Informasi artikel
Isi pada bagian `ARTIKEL` di `app.py` (penulis, jurnal, DOI) bila ingin ditampilkan di Beranda.

## Struktur proyek
```text
.
├── app.py             # antarmuka Streamlit (6 halaman)
├── pipeline.py        # logika penelitian (dari notebook)
├── plots.py           # seluruh gambar notebook sebagai fungsi
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── README.md
    └── nasa_power_bima_raw.csv   # <-- WAJIB Anda letakkan (dari results/ di Colab)
```

## Halaman aplikasi
1. **Beranda** – judul, tujuan, dataset, metode, parameter, lingkungan komputasi.
2. **Dataset** – data mentah, kualitas data, statistik & distribusi, korelasi, transformasi.
3. **Klasterisasi** – penentuan K, hasil 4 algoritma, PCA & dendrogram, ambiguitas GMM.
4. **Evaluasi & Perbandingan** – metrik internal, ARI/NMI, silhouette plot, skor gabungan, bootstrap.
5. **Validasi Musiman** – Chi-square, Cramér's V, ARI/NMI terhadap musim, heatmap komposisi.
6. **Interpretasi & Hasil** – ringkasan, karakteristik klaster, pola temporal, uji beda, keterbatasan,
   dan **cek reproduksi** terhadap output notebook.

## Menjalankan secara lokal
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Gunakan Python 3.11–3.13 (versi `scipy==1.16.3` membutuhkan Python ≥ 3.11).
Klik **▶️ Jalankan analisis** di sidebar. Eksekusi pertama ±1–3 menit (SOM dan bootstrap dihitung
dengan Python murni/NumPy); hasil di-cache sehingga eksekusi berikutnya cepat.

## Deploy ke Streamlit Community Cloud
1. Pastikan `data/nasa_power_bima_raw.csv` sudah ada, lalu push seluruh folder ke repositori GitHub.
2. Buka https://share.streamlit.io → **Create app** → pilih repositori, branch, dan file utama `app.py`.
3. Pada **Advanced settings**, pilih Python **3.12** (atau 3.13).
4. Klik **Deploy**.

## Sumber data
- **Dataset bawaan penelitian** (disarankan): berkas CSV pada `data/`.
- **Unduh ulang dari NASA POWER API**: parameter sama dengan penelitian. Arsip NASA dapat diperbarui
  sehingga angka dapat sedikit berbeda.
- **Unggah CSV**: kolom `date, suhu_c, kelembapan_persen, curah_hujan_mm, angin_ms, tekanan_kpa, radiasi_mj_m2`.

## Reproduksibilitas
Versi library dipin sama dengan notebook (NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.16.3,
matplotlib 3.10.0, seaborn 0.13.2). Tab **Cek reproduksi** membandingkan hasil aplikasi dengan angka
pada notebook penelitian.

## Penamaan klaster
Nama klaster (mis. "Monsun basah") ditetapkan manual oleh peneliti pada notebook. Aplikasi hanya
menerapkannya bila sebaran klaster K-Means identik dengan penelitian (K=3; 1379/1990/1014 hari);
selain itu klaster diberi nama generik agar tidak keliru.
