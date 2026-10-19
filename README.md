# Clustering Algorithm Comparison and Seasonal Validation for Daily Weather Patterns in Bima Regency

Aplikasi web **Streamlit** untuk mendemonstrasikan penelitian yang telah dipublikasikan:
perbandingan empat algoritma klasterisasi (K-Means, Gaussian Mixture Model, Self-Organizing Map,
Hierarchical/Ward) pada data cuaca harian NASA POWER Kabupaten Bima, NTB (2014–2025), beserta
validasi eksternal terhadap musim.

## Struktur proyek
```text
.
├── app.py             
├── pipeline.py       
├── plots.py           
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── README.md
    └── nasa_power_bima_raw.csv   
```

## Halaman aplikasi
1. **Beranda** – judul, tujuan, dataset, metode, parameter, lingkungan komputasi.
2. **Dataset** – data mentah, kualitas data, statistik & distribusi, korelasi, transformasi.
3. **Klasterisasi** – penentuan K, hasil 4 algoritma, PCA & dendrogram, ambiguitas GMM.
4. **Evaluasi & Perbandingan** – metrik internal, ARI/NMI, silhouette plot, skor gabungan, bootstrap.
5. **Validasi Musiman** – Chi-square, Cramér's V, ARI/NMI terhadap musim, heatmap komposisi.
6. **Interpretasi & Hasil** – ringkasan, karakteristik klaster, pola temporal, uji beda, keterbatasan,
   dan **cek reproduksi** terhadap output notebook.

## Penamaan klaster
Nama klaster (mis. "Monsun basah") ditetapkan manual oleh peneliti pada notebook. Aplikasi hanya
menerapkannya bila sebaran klaster K-Means identik dengan penelitian (K=3; 1379/1990/1014 hari);
selain itu klaster diberi nama generik agar tidak keliru.
