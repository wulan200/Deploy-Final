"""
app.py — Aplikasi Streamlit
"Clustering Algorithm Comparison and Seasonal Validation for Daily Weather
Patterns in Bima Regency"

"""
import os
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import streamlit as st

import pipeline as P
import plots as PL

st.set_page_config(page_title="Pola Cuaca Harian Bima", page_icon="🌦️", layout="wide")

# ---------------------------------------------------------------------------
# INFORMASI ARTIKEL — ISI SENDIRI (tidak terdapat pada kode penelitian)
# Biarkan "" bila tidak ingin ditampilkan.
# ---------------------------------------------------------------------------
ARTIKEL = {
    "penulis": "",      # contoh: "Nama Penulis, dkk."
    "jurnal": "",       # contoh: "Nama Jurnal, Vol. X No. Y, 2026"
    "doi": "",          # contoh: "https://doi.org/..."
}

JUDUL_EN = "Clustering Algorithm Comparison and Seasonal Validation for Daily Weather Patterns in Bima Regency"
JUDUL_ID = ("Perbandingan Algoritma Klasterisasi untuk Identifikasi Pola Cuaca Harian "
            "Berbasis Data NASA POWER di Kabupaten Bima, Nusa Tenggara Barat")

_BASE = os.path.dirname(os.path.abspath(__file__))
_NAMA_CSV = "nasa_power_bima_raw.csv"
DATA_PATH = next((p for p in (os.path.join(_BASE, "data", _NAMA_CSV), os.path.join(_BASE, _NAMA_CSV))
                  if os.path.exists(p)), os.path.join(_BASE, "data", _NAMA_CSV))
LABEL_DATA = "Dataset bawaan penelitian (NASA POWER, 2014–2025)"

HALAMAN = ["🏠 Beranda", "🗂️ Dataset", "🧩 Klasterisasi", "📊 Evaluasi & Perbandingan",
           "🍂 Validasi Musiman", "📝 Interpretasi & Hasil"]


# ---------------------------------------------------------------------------
# FUNGSI BANTU
# ---------------------------------------------------------------------------
def show(fig):
    """Menampilkan figure matplotlib lalu menutupnya (hindari kebocoran memori)."""
    if fig is None:
        return
    st.pyplot(fig)
    plt.close(fig)


def tampil(df):
    """Format tanggal tanpa jam untuk tampilan tabel."""
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"]).dt.date
    return d


@st.cache_data(show_spinner=False)
def load_bundled(path):
    """Membaca dataset bawaan penelitian dan memvalidasi kolomnya (tanpa pengambilan data baru)."""
    return P.standardize_raw(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def cached_run(df):
    return P.run_pipeline(df)


def set_data(df, label):
    st.session_state["df_raw"] = df
    st.session_state["label_data"] = label
    st.session_state["R"] = None


def jalankan_analisis():
    if st.session_state.get("df_raw") is None:
        st.error("Dataset bawaan tidak ditemukan. Letakkan data/nasa_power_bima_raw.csv.")
        return
    with st.spinner("Menjalankan seluruh alur penelitian (±1–3 menit pada eksekusi pertama)..."):
        try:
            st.session_state["R"] = cached_run(st.session_state["df_raw"])
        except Exception as e:  # tampilkan pesan yang jelas, bukan crash
            st.session_state["R"] = None
            st.error(f"Analisis gagal: {e}")


def butuh_hasil():
    """Mengembalikan dict hasil; bila belum ada, tampilkan tombol lalu hentikan halaman."""
    R = st.session_state.get("R")
    if R is None:
        st.info("Analisis belum dijalankan. Klik tombol di bawah atau di sidebar.")
        if st.button("▶️ Jalankan analisis sekarang", type="primary", key="run_gate"):
            jalankan_analisis()
            st.rerun()
        st.stop()
    return R


def gate(R):
    """Untuk tab: tampilkan info bila hasil belum ada, tanpa menghentikan halaman."""
    if R is None:
        st.info("Klik **Jalankan analisis** di sidebar untuk menampilkan bagian ini.")
        return False
    return True


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("🌦️ Pola Cuaca Bima")
halaman = st.sidebar.radio("Navigasi", HALAMAN, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.subheader("Dataset")

if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
    st.session_state["label_data"] = None
    st.session_state["R"] = None

if st.session_state["df_raw"] is None:  # muat dataset bawaan secara otomatis
    if os.path.exists(DATA_PATH):
        try:
            set_data(load_bundled(DATA_PATH), LABEL_DATA)
        except Exception as e:
            st.sidebar.error(f"Dataset bawaan tidak valid: {e}")
    else:
        st.sidebar.warning("Berkas data/nasa_power_bima_raw.csv tidak ditemukan. Lihat README.")

if st.session_state["df_raw"] is not None:
    st.sidebar.success(f"Data aktif: {st.session_state['label_data']} "
                       f"({len(st.session_state['df_raw'])} baris)")
else:
    st.sidebar.info("Dataset belum termuat.")

if st.sidebar.button("▶️ Jalankan analisis", type="primary"):
    jalankan_analisis()

if st.session_state.get("R") is not None:
    st.sidebar.caption("✅ Hasil analisis tersedia.")


# ---------------------------------------------------------------------------
# HALAMAN: BERANDA
# ---------------------------------------------------------------------------
def page_beranda():
    st.title(JUDUL_EN)
    st.caption(f"Judul pada notebook penelitian: {JUDUL_ID}")
    info = [f"**{k.capitalize()}**: {v}" for k, v in ARTIKEL.items() if v]
    if info:
        st.markdown("  \n".join(info))

    st.markdown(
        "Aplikasi ini adalah **deployment interaktif dari penelitian yang telah dipublikasikan**. "
        "Seluruh metode, parameter, dan metrik dipertahankan sama seperti pada kode penelitian; "
        "aplikasi tidak memperkenalkan analisis baru."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tujuan aplikasi")
        st.markdown(
            "- Mendemonstrasikan perbandingan empat algoritma klasterisasi pada data cuaca harian.\n"
            "- Menampilkan validasi hasil klaster terhadap pembagian musim (validasi eksternal).\n"
            "- Memungkinkan hasil penelitian ditelusuri langkah demi langkah dari data hingga interpretasi."
        )
        st.subheader("Metode yang digunakan")
        st.markdown(
            "1. Pengambilan data harian NASA POWER dan pra-pemrosesan (penanganan nilai hilang, duplikasi, "
            "validasi rentang fisik, interpolasi linear berbasis waktu).\n"
            "2. Analisis korelasi, transformasi `log1p` (skenario B) dibanding tanpa transformasi (skenario A), "
            "lalu standardisasi.\n"
            "3. Penentuan jumlah klaster K (Elbow kuantitatif, Silhouette, BIC GMM, skor gabungan).\n"
            "4. Klasterisasi dengan **K-Means, GMM, SOM, dan Hierarchical (Ward)**.\n"
            "5. Evaluasi internal (Silhouette, Calinski-Harabasz, Davies-Bouldin) dan kesepakatan antaralgoritma (ARI, NMI).\n"
            "6. Validasi eksternal terhadap musim (Chi-square, Cramér's V, ARI, NMI).\n"
            "7. Pemilihan algoritma utama (bobot 0,4 metrik internal + 0,6 validasi eksternal) dan uji stabilitas bootstrap.\n"
            "8. Karakterisasi klaster (z-score), pola temporal, serta uji beda antarklaster."
        )
    with c2:
        st.subheader("Informasi dataset")
        st.markdown(
            f"- **Sumber**: NASA POWER (API harian titik grid, community `AG`)\n"
            f"- **Lokasi**: Kabupaten Bima, Nusa Tenggara Barat (lat {P.LATITUDE}, lon {P.LONGITUDE})\n"
            f"- **Periode**: {P.START_DATE[:4]}-{P.START_DATE[4:6]}-{P.START_DATE[6:]} s/d "
            f"{P.END_DATE[:4]}-{P.END_DATE[4:6]}-{P.END_DATE[6:]} (12 tahun kalender utuh)\n"
            f"- **Resolusi**: harian"
        )
        st.dataframe(pd.DataFrame({
            "Parameter NASA POWER": P.NASA_PARAMETERS,
            "Nama kolom": P.FEATURE_COLS_EXPECTED,
        }), hide_index=True)
        st.subheader("Algoritma yang dibandingkan")
        st.dataframe(pd.DataFrame({
            "Algoritma": ["K-Means", "Gaussian Mixture Model", "Self-Organizing Map", "Hierarchical (Ward)"],
            "Asumsi": ["Klaster bulat, ukuran seimbang, keanggotaan tegas",
                       "Klaster Gaussian, keanggotaan probabilistik",
                       "Mempertahankan hubungan topologis",
                       "Struktur berjenjang"],
        }), hide_index=True)

    with st.expander("Parameter penelitian (tetap, sesuai kode)"):
        st.dataframe(pd.DataFrame([
            ("Random state", P.RANDOM_STATE), ("Rentang K diuji", f"{P.K_MIN}–{P.K_MAX}"),
            ("Alpha uji statistik", P.ALPHA), ("Ambang korelasi pantau / keluarkan", f"{P.CORR_WATCH} / {P.CORR_DROP}"),
            ("Ambang |skewness| untuk transformasi", P.SKEW_THRESHOLD),
            ("Ambang z-score (tinggi/rendah)", P.Z_THRESHOLD),
            ("Bootstrap", f"{P.N_BOOTSTRAP} iterasi, subsampel {int(P.SUBSAMPLE_FRAC * 100)}%"),
            ("Ambang hari ambigu (GMM)", P.AMBANG_AMBIGU),
            ("Bobot skor akhir (internal / eksternal)", f"{P.BOBOT_INTERNAL} / {P.BOBOT_EKSTERNAL}"),
            ("K-Means", "n_init=10"), ("GMM", "covariance_type='full', n_init=10"),
            ("SOM", "grid persegi, ~5·√n node, 12.000 iterasi, lr0=0,5, tetangga Gaussian; node dikelompokkan K-Means"),
            ("Hierarchical", "AgglomerativeClustering, linkage='ward'"),
        ], columns=["Parameter", "Nilai"]).astype(str), hide_index=True)

    with st.expander("Lingkungan komputasi"):
        st.dataframe(pd.DataFrame({
            "Komponen": ["Python", "NumPy", "pandas", "scikit-learn", "SciPy", "matplotlib", "seaborn"],
            "Saat penelitian (notebook)": ["3.13.15", "2.1.3", "2.2.3", "1.6.1", "1.16.3", "3.10.0", "0.13.2"],
            "Runtime aplikasi ini": [platform.python_version(), np.__version__, pd.__version__,
                                     sklearn.__version__, scipy.__version__, matplotlib.__version__,
                                     sns.__version__],
        }), hide_index=True)
        st.caption("Data NASA POWER pada penelitian diakses 2026-10-05 05:28. Perbedaan versi library "
                   "dapat memengaruhi hasil numerik (mis. label klaster); lihat tab "
                   "‘Cek reproduksi’ pada halaman Interpretasi & Hasil.")

    st.info("Mulai dari sidebar: klik **Jalankan analisis** untuk memproses dataset bawaan penelitian.")


# ---------------------------------------------------------------------------
# HALAMAN: DATASET
# ---------------------------------------------------------------------------
def page_dataset():
    st.header("Dataset")
    df = st.session_state.get("df_raw")
    R = st.session_state.get("R")
    if df is None:
        st.warning("Dataset belum termuat. Letakkan `data/nasa_power_bima_raw.csv` lalu muat ulang halaman.")
        return
    st.caption(f"Dataset: **{st.session_state['label_data']}**")

    t1, t2, t3, t4, t5 = st.tabs(["Data mentah", "Kualitas data", "Statistik & distribusi",
                                  "Korelasi", "Transformasi & standardisasi"])
    with t1:
        fc = [c for c in df.columns if c != "date"]
        m = st.columns(5)
        m[0].metric("Jumlah data (baris)", f"{len(df):,}")
        m[1].metric("Jumlah variabel", len(fc))
        m[2].metric("Awal", str(df["date"].min().date()))
        m[3].metric("Akhir", str(df["date"].max().date()))
        m[4].metric("Jumlah tahun", df["date"].dt.year.nunique())
        st.dataframe(tampil(df), hide_index=True)
        st.markdown("**Informasi dasar**")
        st.dataframe(pd.DataFrame({"tipe data": df.dtypes.astype(str),
                                   "nilai kosong": df.isna().sum()}))
        if R is not None:
            st.markdown("**Pemeriksaan ketersediaan parameter** (NASA POWER memakai −999 sebagai penanda data tidak tersedia)")
            st.dataframe(R["cek_parameter"], hide_index=True)
            if R["cek_parameter"]["jumlah_-999"].sum() == 0:
                st.success("Seluruh parameter tersedia penuh untuk periode yang dipilih.")
            else:
                st.warning("Terdapat penanda −999 pada data.")
    with t2:
        if gate(R):
            st.markdown("**Tabel kualitas data (hasil pra-pemrosesan)**")
            st.dataframe(R["audit_df"], hide_index=True)
            st.markdown("**Nilai di luar rentang fisik per variabel**")
            st.dataframe(pd.DataFrame({"variabel": list(R["invalid_report"]),
                                       "jumlah": list(R["invalid_report"].values())}), hide_index=True)
            with st.expander("Tahapan pra-pemrosesan sesuai kode"):
                st.markdown(
                    "1. Nilai −999 → NaN\n2. Hapus tanggal duplikat\n3. Lengkapi deret tanggal harian (reindex)\n"
                    "4. Konversi numerik\n5. Validasi rentang fisik (suhu −10–55 °C; kelembapan 0–100 %; hujan, angin, radiasi ≥ 0; "
                    "tekanan 50–110 kPa) → di luar rentang menjadi NaN\n6. Imputasi interpolasi linear berbasis waktu\n\n"
                    "Outlier meteorologis (mis. hujan ekstrem) **tidak dihapus** karena merupakan kejadian fisik yang valid.")
    with t3:
        if gate(R):
            fc, dc = R["feature_cols"], R["df_clean"]
            st.markdown("**Statistik deskriptif**")
            st.dataframe(R["desc"])
            if R["menceng"]:
                st.write(f"Variabel dengan |skewness| > {P.SKEW_THRESHOLD}: **{R['menceng']}**. "
                         "Variabel semacam ini berpotensi mendominasi perhitungan jarak Euclidean.")
            a, b, c = st.tabs(["Histogram", "Boxplot", "Profil rata-rata bulanan"])
            with a:
                show(PL.fig_histogram(dc, fc))
            with b:
                show(PL.fig_boxplot(dc, fc))
            with c:
                show(PL.fig_profil_bulanan(R["profil_bulanan"], fc))
                st.dataframe(R["profil_bulanan"])
    with t4:
        if gate(R):
            show(PL.fig_korelasi(R["corr"]))
            pas = R["pasangan_korelasi"]
            st.write(f"Pasangan |r| ≥ {P.CORR_WATCH}: " +
                     (str([(a, b, round(r, 3)) for a, b, r in pas]) if pas else "tidak ada"))
            st.write(f"Dikeluarkan (|r| ≥ {P.CORR_DROP}): " + (str(R["dibuang"]) if R["dibuang"] else "tidak ada"))
            st.success(f"Fitur final ({len(R['features_final'])}): {R['features_final']}")
    with t5:
        if gate(R):
            if len(R["log_trans"]):
                st.markdown("**Variabel yang ditransformasi (log1p)**")
                st.dataframe(R["log_trans"], hide_index=True)
                show(PL.fig_transformasi(R["df_clean"], R["df_trans"], R["kandidat"]))
            else:
                st.write("Tidak ada variabel yang memerlukan transformasi.")
            st.markdown("**Rentang nilai terstandardisasi** (A: tanpa transformasi, B: dengan log1p)")
            st.dataframe(R["rentang_standar"])
            st.caption("Nilai maksimum yang lebih kecil pada skenario B menandakan dominasi hari ekstrem "
                       "terhadap perhitungan jarak telah berkurang.")


# ---------------------------------------------------------------------------
# HALAMAN: KLASTERISASI
# ---------------------------------------------------------------------------
def page_klaster():
    st.header("Klasterisasi")
    R = butuh_hasil()
    best_k = R["best_k"]
    t1, t2, t3, t4 = st.tabs(["Penentuan jumlah klaster (K)", "Hasil empat algoritma",
                              "Visualisasi (PCA & dendrogram)", "Keanggotaan probabilistik GMM"])
    with t1:
        st.markdown("**Perbandingan skenario pra-pemrosesan**")
        st.dataframe(R["banding_skenario"], hide_index=True)
        st.write(f"Silhouette terbaik A: **{R['sil_best_A']:.4f}** (K={R['kA']}) · "
                 f"Silhouette terbaik B: **{R['sil_best_B']:.4f}** (K={R['kB']})")
        st.success(f"Skenario terpilih: **{R['SKENARIO']}**")
        st.caption("Ambang −0,01 dipakai karena bila selisih silhouette sangat tipis, skenario dengan "
                   "transformasi tetap lebih layak: lebih memenuhi asumsi jarak Euclidean dan lebih tahan "
                   "terhadap pengaruh hari ekstrem.")
        show(PL.fig_penentuan_k(R["metrics_df"], R["k_elbow"], R["k_sil"], R["k_bic"]))
        m = st.columns(4)
        m[0].metric("Elbow kuantitatif", f"K = {R['k_elbow']}")
        m[1].metric("Silhouette maks", f"K = {R['k_sil']}")
        m[2].metric("BIC minimum (GMM)", f"K = {R['k_bic']}")
        m[3].metric("Skor gabungan (dipakai)", f"K = {best_k}")
        st.markdown("**Tabel evaluasi jumlah klaster**")
        st.dataframe(R["metrics_df"].round(4), hide_index=True)
        if len({R["k_elbow"], R["k_sil"], R["k_bic"], best_k}) == 1:
            st.write("Seluruh kriteria konvergen pada nilai K yang sama.")
        else:
            st.write("Kriteria tidak seluruhnya konvergen. Skor gabungan dipakai sebagai penengah karena "
                     "mempertimbangkan kohesi, separasi, dan rasio dispersi sekaligus (Silhouette + "
                     "Calinski-Harabasz + kebalikan Davies-Bouldin, masing-masing dinormalisasi min–max).")
    with t2:
        st.markdown(f"**Sebaran anggota klaster (K = {best_k})**")
        st.dataframe(R["sebaran_klaster"])
        si = R["som_info"]
        st.caption(f"SOM: grid {si['sisi']}×{si['sisi']} = {si['n_node']} node (target ≈ {si['target']:.0f}); "
                   f"quantization error {si['quantization_error']:.4f}; node terpakai {si['node_terpakai']} "
                   f"dari {si['n_node']}. Tingkat kedua: node dikelompokkan dengan K-Means.")
        alg = st.selectbox("Lihat label harian untuk algoritma", list(R["hasil_label"]),
                           index=list(R["hasil_label"]).index(R["ALG_TERBAIK"]))
        lb = R["hasil_label"][alg]
        cnt = pd.Series(lb).value_counts().sort_index()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.bar_chart(cnt.rename("jumlah hari"))
        with c2:
            st.dataframe(pd.DataFrame({"date": R["df_clean"]["date"].dt.date, "klaster": lb}),
                         hide_index=True, height=260)
    with t3:
        show(PL.fig_pca(R["P"], R["ev"], R["hasil_label"], R["sil_score"], best_k))
        st.caption(f"PC1 menjelaskan {R['ev'][0]:.2f}% varians, PC2 {R['ev'][1]:.2f}%. Total {R['ev'].sum():.2f}%. "
                   f"PCA dipakai HANYA untuk visualisasi; klasterisasi tetap dilakukan pada seluruh "
                   f"{len(R['features_final'])} dimensi variabel.")
        st.markdown("**Loading PCA**")
        st.dataframe(R["loading"])
        show(PL.fig_dendrogram(R["linkage_Z"]))
        st.caption("Dendrogram dipangkas pada 30 node teratas karena dendrogram penuh tidak terbaca pada data berukuran besar.")
    with t4:
        n = len(R["proba_maks"])
        m = st.columns(3)
        m[0].metric("Rata-rata probabilitas tertinggi", f"{R['proba_maks'].mean():.4f}")
        m[1].metric(f"Hari dengan probabilitas < {P.AMBANG_AMBIGU}", R["n_ambigu"])
        m[2].metric("Persentase hari ambigu", f"{R['n_ambigu'] / n * 100:.2f}%")
        show(PL.fig_ambiguitas(R["proba_maks"], R["ambigu_per_bulan"]))
        pb = R["ambigu_per_bulan"]
        st.write(f"Bulan dengan hari peralihan terbanyak: **{int(pb.idxmax())}** ({pb.max():.1f}%).")


# ---------------------------------------------------------------------------
# HALAMAN: EVALUASI & PERBANDINGAN
# ---------------------------------------------------------------------------
def page_evaluasi():
    st.header("Evaluasi & Perbandingan Algoritma")
    R = butuh_hasil()
    t1, t2, t3, t4, t5 = st.tabs(["Metrik internal", "Kesepakatan antaralgoritma", "Silhouette plot",
                                  "Penilaian gabungan", "Stabilitas bootstrap"])
    with t1:
        pb = R["perbandingan"]
        sty = (pb.style.format({"Silhouette": "{:.4f}", "Calinski_Harabasz": "{:.4f}",
                                "Davies_Bouldin": "{:.4f}", "Klaster_terkecil_persen": "{:.4f}"})
               .highlight_max(subset=["Silhouette", "Calinski_Harabasz"], color="#c6efce")
               .highlight_min(subset=["Davies_Bouldin"], color="#c6efce"))
        st.dataframe(sty, hide_index=True)
        st.caption("Hijau = nilai terbaik per metrik (Silhouette & Calinski-Harabasz: makin tinggi makin baik; "
                   "Davies-Bouldin: makin rendah makin baik).")
        show(PL.fig_perbandingan_metrik(pb))
    with t2:
        show(PL.fig_kesepakatan(R["ari_mat"], R["nmi_mat"]))
        st.write(f"Rata-rata ARI antaralgoritma: **{R['ari_antar_mean']:.4f}** · "
                 f"Rentang: {R['ari_antar_min']:.4f} – {R['ari_antar_max']:.4f}")
        st.caption("Kesepakatan yang tinggi menunjukkan struktur klaster memang ada dalam data, "
                   "bukan artefak dari satu algoritma tertentu.")
    with t3:
        show(PL.fig_silhouette(R["hasil_label"], R["sil_samples"], R["sil_score"], R["best_k"]))
    with t4:
        st.markdown(f"Bobot metrik internal **{P.BOBOT_INTERNAL}**, validasi eksternal (Cramér's V) **{P.BOBOT_EKSTERNAL}**")
        st.dataframe(R["skor"].round(4), hide_index=True)
        st.success(f"Algoritma utama yang dipilih: **{R['ALG_TERBAIK']}**")
    with t5:
        st.markdown(f"**Uji stabilitas {R['ALG_TERBAIK']}** ({P.N_BOOTSTRAP} iterasi, subsampel {int(P.SUBSAMPLE_FRAC * 100)}%)")
        s = R["stab"]
        m = st.columns(3)
        m[0].metric("ARI terhadap model penuh", f"{R['ari_mean']:.4f}", f"sd = {R['ari_std']:.4f}", delta_color="off")
        m[1].metric("Rentang ARI", f"{s['ari_vs_model_penuh'].min():.4f} – {s['ari_vs_model_penuh'].max():.4f}")
        m[2].metric("Silhouette subsampel", f"{s['silhouette'].mean():.4f}", f"sd = {s['silhouette'].std():.4f}", delta_color="off")
        st.write(f"Interpretasi: {R['tafsir_stab']}.")
        show(PL.fig_stabilitas(s, R["ari_mean"], R["sil_final"]))
        with st.expander("Tabel per iterasi"):
            st.dataframe(s.round(4), hide_index=True)


# ---------------------------------------------------------------------------
# HALAMAN: VALIDASI MUSIMAN
# ---------------------------------------------------------------------------
def page_musiman():
    st.header("Validasi Eksternal terhadap Musim")
    R = butuh_hasil()
    st.markdown(
        "Algoritma **tidak pernah menerima informasi waktu**. Label musim hanya dipakai sesudah klasterisasi "
        "sebagai pembanding eksternal."
    )
    st.markdown("**Definisi musim referensi (sesuai kode, berbasis bulan kalender)**")
    st.dataframe(pd.DataFrame({
        "Musim": P.URUT_MUSIM,
        "Bulan": ["Des, Jan, Feb, Mar", "Apr, Mei", "Jun, Jul, Ags, Sep", "Okt, Nov"],
    }), hide_index=True)

    st.markdown("**Tabel validasi eksternal**")
    tv = R["tabel_validasi"]
    st.dataframe(tv.style.highlight_max(subset=["Cramers_V"], color="#c6efce"), hide_index=True)
    a = R["alg_terbaik_musim"]
    st.success(f"Keselarasan tertinggi dengan musim: **{a}** (Cramér's V = {R['v_maks']:.4f}, {P.tafsir_v(R['v_maks'])})")
    st.caption("Interpretasi Cramér's V pada kode: ≥0,5 sangat kuat; ≥0,3 kuat; ≥0,1 sedang; selain itu lemah. "
               "Algoritma terpilih akhir ditetapkan lewat skor gabungan (halaman Evaluasi), "
               "bukan hanya berdasarkan Cramér's V.")
    show(PL.fig_validasi_musim(R["hasil_validasi"]))
    with st.expander("Tabel kontingensi musim × klaster"):
        alg = st.selectbox("Algoritma", list(R["hasil_validasi"]), key="kont_alg")
        st.dataframe(R["hasil_validasi"][alg]["kontingensi"].rename_axis(index="Musim", columns="Klaster"))


# ---------------------------------------------------------------------------
# HALAMAN: INTERPRETASI & HASIL
# ---------------------------------------------------------------------------
def page_hasil():
    st.header("Interpretasi & Hasil")
    R = butuh_hasil()
    names, k = R["cluster_names"], R["best_k"]
    t1, t2, t3, t4, t5, t6 = st.tabs(["Ringkasan", "Karakteristik klaster", "Pola temporal",
                                      "Uji beda antarklaster", "Keterbatasan", "Cek reproduksi"])
    with t1:
        dc = R["df_clustered"]
        m = st.columns(4)
        m[0].metric("Observasi harian", f"{len(dc):,}")
        m[1].metric("Variabel klasterisasi", len(R["features_final"]))
        m[2].metric("K digunakan", k)
        m[3].metric("Algoritma terpilih", R["ALG_TERBAIK"])
        m = st.columns(4)
        m[0].metric("Silhouette", f"{R['sil_final']:.4f}")
        cv = R["hasil_validasi"][R["ALG_TERBAIK"]]["cramers_v"]
        m[1].metric("Cramér's V vs musim", f"{cv:.4f}", P.tafsir_v(cv), delta_color="off")
        m[2].metric("Stabilitas (ARI)", f"{R['ari_mean']:.4f}", f"sd = {R['ari_std']:.4f}", delta_color="off")
        m[3].metric("Hari ambigu (GMM)", R["n_ambigu"], f"{R['n_ambigu'] / len(dc) * 100:.2f}%", delta_color="off")
        st.markdown(
            f"- **Data**: Kabupaten Bima, NTB ({P.LATITUDE}, {P.LONGITUDE}); {dc['date'].min().date()} s/d "
            f"{dc['date'].max().date()}; kelengkapan data {R['audit']['kelengkapan_persen']}%.\n"
            f"- **Pra-pemrosesan**: skenario terpilih {R['SKENARIO']}; ditransformasi: {R['kandidat'] or 'tidak ada'}.\n"
            f"- **Penentuan K**: Elbow={R['k_elbow']} | Silhouette={R['k_sil']} | BIC={R['k_bic']} | "
            f"Gabungan={k} (dipakai).\n"
            f"- **Algoritma terpilih**: {R['ALG_TERBAIK']} (skor akhir tertinggi pada penilaian gabungan).\n"
            f"- **Stabilitas**: {R['tafsir_stab']}."
        )
        show(PL.fig_distribusi_centroid(R["cluster_counts"], R["cluster_pct"], R["centroid"],
                                        R["ALG_TERBAIK"], k))
        show(PL.fig_radar(R["centroid"], names, R["features_final"], k))
        dom = int(R["cluster_counts"].idxmax())
        st.write(f"Klaster paling dominan: **Klaster {dom}** ({names[dom]}, {R['cluster_pct'][dom]:.2f}%)")
        st.download_button("⬇️ Unduh hasil klasterisasi lengkap (CSV)",
                           dc.to_csv(index=False).encode("utf-8"),
                           "hasil_klasterisasi_lengkap.csv", "text/csv")
    with t2:
        if not R["nama_berlaku"]:
            st.info("Penamaan klaster pada penelitian ditetapkan **manual** oleh peneliti. Karena sebaran klaster "
                    "run ini tidak identik dengan penelitian (atau data/K berbeda), nama manual tidak diterapkan "
                    "dan klaster diberi nama generik.")
        else:
            st.caption("Nama klaster ditetapkan manual oleh peneliti berdasarkan profil z-score dan pola bulanan.")
        st.markdown("**Rata-rata tiap variabel per klaster (skala asli)**")
        st.dataframe(R["cluster_mean"])
        show(PL.fig_zscore(R["z_table"], k))
        for c in sorted(names):
            if c not in R["cluster_counts"].index:
                continue
            bd = ", ".join(R["bulanan_pct"][c].sort_values(ascending=False).head(3).index)
            md = R["hasil_validasi"][R["ALG_TERBAIK"]]["kontingensi"][c].idxmax()
            with st.expander(f"Klaster {c} — {names[c]} ({R['cluster_counts'][c]} hari, {R['cluster_pct'][c]:.2f}%)"):
                st.write(f"Bulan dominan: **{bd}** · Musim dominan: **{md}**")
                for b in R["interpretasi"][c]:
                    st.markdown(f"- {b}")
        st.caption(f"Kriteria pada kode: z > {P.Z_THRESHOLD} ‘relatif tinggi’; z < −{P.Z_THRESHOLD} ‘relatif rendah’; "
                   "selain itu ‘mendekati rata-rata’.")
    with t3:
        st.markdown("**Distribusi bulanan**")
        show(PL.fig_pola_bulanan(R["bulanan_pct"]))
        st.dataframe(R["bulanan"])
        st.markdown("**Variasi antartahun**")
        show(PL.fig_pola_tahunan(R["tahunan_pct"], names))
        st.dataframe(R["tahunan_pct"])
        st.caption("Catatan pada kode penelitian: rentang 12 tahun tidak memadai untuk menyimpulkan tren perubahan "
                   "iklim; variasi antartahun lebih tepat dikaitkan dengan variabilitas alami seperti ENSO dan "
                   "Indian Ocean Dipole.")
    with t4:
        st.markdown("**Perbedaan variabel antarklaster (deskriptif)**")
        st.dataframe(R["uji_df"], hide_index=True)
        top = R["uji_df"].sort_values("effect_size", ascending=False).head(3)
        st.write("Variabel dengan daya pembeda terkuat: " +
                 "; ".join(f"**{r['variabel']}** (effect size {r['effect_size']:.3f}, {r['kekuatan']})"
                           for _, r in top.iterrows()))
        st.markdown("**Post-hoc Dunn (koreksi Bonferroni)**")
        st.dataframe(R["posthoc"], hide_index=True)
        st.markdown("**Ringkasan per pasangan klaster**")
        st.dataframe(R["ringkas_posthoc"].rename("ringkasan"))
        if len(R["pasangan_lemah"]):
            st.warning("Pasangan klaster berikut kurang terpisah dengan baik (≤ 2 variabel berbeda signifikan):")
            st.dataframe(R["pasangan_lemah"].rename("ringkasan"))
        else:
            st.write("Tidak ada pasangan klaster dengan ≤ 2 variabel yang berbeda signifikan. "
                     "Pasangan dengan variabel yang tidak berbeda signifikan dapat dilihat pada tabel di atas.")
        st.caption("Uji dilakukan pada variabel yang sama yang membentuk klaster sehingga bersifat deskriptif, "
                   "bukan uji hipotesis konfirmatori.")
    with t5:
        st.markdown("**Keterbatasan penelitian (sesuai notebook penelitian; mengacu pada dataset penelitian)**")
        st.markdown(
            "1. **Autokorelasi temporal.** Setiap hari diperlakukan sebagai observasi independen, padahal kondisi cuaca hari berurutan sangat berkorelasi. Jumlah observasi efektif karena itu lebih kecil daripada 4.383 baris data, sehingga uji statistik cenderung terlalu mudah menghasilkan kesimpulan signifikan.\n"
            "2. **Sirkularitas uji antarklaster.** Uji perbedaan dilakukan pada variabel yang sama yang membentuk klaster, sehingga bersifat deskriptif untuk mengukur besarnya pemisahan, bukan uji hipotesis konfirmatori.\n"
            "3. **Resolusi data reanalysis.** NASA POWER merepresentasikan kondisi rata-rata sel grid sekitar 0,5 derajat, bukan pengukuran titik. Fenomena cuaca lokal berskala kecil dan pengaruh topografi tidak sepenuhnya tertangkap.\n"
            "4. **Panjang periode data.** Rentang 12 tahun memadai untuk mengidentifikasi pola musiman, tetapi berada di bawah standar WMO 30 tahun untuk penetapan normal iklim, sehingga tidak dapat dipakai menyimpulkan tren perubahan iklim.\n"
            "5. **Batas musim yang disederhanakan.** Validasi eksternal memakai pembagian musim berbasis bulan kalender, sedangkan awal musim sesungguhnya bervariasi antartahun menurut Zona Musim BMKG untuk Kabupaten Bima.\n"
            "6. **Penetapan jumlah klaster.** Kriteria penentuan K tidak seluruhnya konvergen (Elbow K=4, Silhouette K=3, BIC K=8). Penetapan K=3 karena itu mengandung unsur keputusan peneliti, meskipun didukung skor gabungan dan keselarasan dengan pembagian musim."
        )
    with t6:
        st.markdown("Perbandingan hasil run ini dengan **output pada notebook penelitian**. "
                    "Relevan bila data yang dipakai adalah dataset penelitian.")
        cek = P.cek_reproduksi(R)
        n_ok = int((cek["Sesuai"] == "✅").sum())
        if n_ok == len(cek):
            st.success(f"Seluruh {len(cek)} item sesuai dengan notebook penelitian.")
        else:
            st.warning(f"{n_ok} dari {len(cek)} item sesuai. Ketidaksesuaian dapat berasal dari perbedaan data "
                       "(mis. pembaruan arsip NASA POWER) atau versi library.")
        st.dataframe(cek, hide_index=True)


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------
{
    HALAMAN[0]: page_beranda,
    HALAMAN[1]: page_dataset,
    HALAMAN[2]: page_klaster,
    HALAMAN[3]: page_evaluasi,
    HALAMAN[4]: page_musiman,
    HALAMAN[5]: page_hasil,
}[halaman]()