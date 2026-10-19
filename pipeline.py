"""
pipeline.py
===========
Logika penelitian "Clustering Algorithm Comparison and Seasonal Validation for
Daily Weather Patterns in Bima Regency", dipindahkan dari notebook Colab.

"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# KONSTANTA PENELITIAN (identik dengan notebook)
# ---------------------------------------------------------------------------
LATITUDE, LONGITUDE = -8.47, 118.75          # Kabupaten Bima, NTB
START_DATE = "20140101"
END_DATE = "20251231"
NASA_PARAMETERS = ["T2M", "RH2M", "PRECTOTCORR", "WS2M", "PS", "ALLSKY_SFC_SW_DWN"]
NAMA_VARIABEL = {
    "T2M": "suhu_c", "RH2M": "kelembapan_persen", "PRECTOTCORR": "curah_hujan_mm",
    "WS2M": "angin_ms", "PS": "tekanan_kpa", "ALLSKY_SFC_SW_DWN": "radiasi_mj_m2",
}
FEATURE_COLS_EXPECTED = [NAMA_VARIABEL[p] for p in NASA_PARAMETERS]

RANDOM_STATE = 42
K_MIN, K_MAX = 2, 8
ALPHA = 0.05
CORR_WATCH, CORR_DROP = 0.85, 0.95
Z_THRESHOLD = 0.5
SKEW_THRESHOLD = 1.0
N_BOOTSTRAP, SUBSAMPLE_FRAC = 50, 0.8
AMBANG_AMBIGU = 0.60
BOBOT_INTERNAL, BOBOT_EKSTERNAL = 0.4, 0.6

URUT_MUSIM = ["Musim Hujan", "Peralihan I", "Musim Kemarau", "Peralihan II"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]

# Penamaan klaster ditetapkan MANUAL oleh peneliti pada notebook (berdasarkan
# profil z-score & pola bulanan hasil penelitian). Disalin apa adanya.
CLUSTER_NAMES_PUBLISHED = {
    0: "Kemarau berangin (monsun timuran)",
    1: "Monsun basah",
    2: "Panas-kering radiasi tinggi (peralihan pra-hujan)",
}
# Sebaran anggota klaster K-Means pada penelitian (output notebook), dipakai
# hanya untuk memastikan penamaan manual di atas cocok dengan label hasil run ini.
CLUSTER_SIZES_PUBLISHED = [1379, 1990, 1014]


# ---------------------------------------------------------------------------
# PEMBAKUAN DATA (dataset bawaan)
# ---------------------------------------------------------------------------
def standardize_raw(df):
    """Memastikan kolom data mentah sesuai format notebook:
    date + 6 variabel (suhu_c, kelembapan_persen, curah_hujan_mm, angin_ms,
    tekanan_kpa, radiasi_mj_m2). Kolom lain diabaikan."""
    if "date" not in df.columns:
        raise ValueError("Kolom 'date' tidak ditemukan.")
    kurang = [c for c in FEATURE_COLS_EXPECTED if c not in df.columns]
    if kurang:
        raise ValueError(f"Kolom berikut tidak ditemukan: {kurang}. "
                         f"Kolom yang dibutuhkan: {['date'] + FEATURE_COLS_EXPECTED}")
    out = df[["date"] + FEATURE_COLS_EXPECTED].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# FUNGSI BANTU (identik notebook)
# ---------------------------------------------------------------------------
def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else pd.Series(np.zeros(len(s)), index=s.index)


def evaluasi_k(X, kmin=K_MIN, kmax=K_MAX, rs=RANDOM_STATE):
    out = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, random_state=rs, n_init=10)
        lb = km.fit_predict(X)
        out.append({"K": k, "Inertia": km.inertia_,
                    "Silhouette": silhouette_score(X, lb),
                    "Calinski_Harabasz": calinski_harabasz_score(X, lb),
                    "Davies_Bouldin": davies_bouldin_score(X, lb)})
    return pd.DataFrame(out)


def cari_elbow(ks, inertias):
    k = np.array(ks, float)
    y = np.array(inertias, float)
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([kn[0], yn[0]]), np.array([kn[-1], yn[-1]])
    v = (p2 - p1) / np.linalg.norm(p2 - p1)
    d = [np.linalg.norm(np.array([kn[i], yn[i]]) - (p1 + np.dot(np.array([kn[i], yn[i]]) - p1, v) * v))
         for i in range(len(kn))]
    return int(k[int(np.argmax(d))])


class SOM:
    """Self-Organizing Map dengan grid persegi dan tetangga Gaussian (identik notebook)."""

    def __init__(self, m, n, dim, n_iter=8000, lr0=0.5, sigma0=None, random_state=42):
        self.m, self.n, self.dim = m, n, dim
        self.n_iter, self.lr0 = n_iter, lr0
        self.sigma0 = sigma0 if sigma0 else max(m, n) / 2.0
        self.rng = np.random.RandomState(random_state)
        self.W = None
        gx, gy = np.meshgrid(np.arange(n), np.arange(m))
        self.grid = np.column_stack([gy.ravel(), gx.ravel()]).astype(float)

    def fit(self, X):
        n_node = self.m * self.n
        idx = self.rng.choice(len(X), n_node, replace=True)
        self.W = X[idx].copy() + self.rng.normal(0, 0.01, (n_node, self.dim))
        for t in range(self.n_iter):
            frac = t / self.n_iter
            lr = self.lr0 * np.exp(-frac * 3)
            sigma = max(self.sigma0 * np.exp(-frac * 3), 0.6)
            x = X[self.rng.randint(len(X))]
            bmu = np.argmin(((self.W - x) ** 2).sum(axis=1))
            d2 = ((self.grid - self.grid[bmu]) ** 2).sum(axis=1)
            h = np.exp(-d2 / (2 * sigma ** 2))[:, None]
            self.W += lr * h * (x - self.W)
        return self

    def bmu(self, X):
        d = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)

    def quantization_error(self, X):
        d = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(axis=2)
        return float(np.sqrt(d.min(axis=1)).mean())


def label_musim(b):
    if b in (12, 1, 2, 3):
        return "Musim Hujan"
    if b in (4, 5):
        return "Peralihan I"
    if b in (6, 7, 8, 9):
        return "Musim Kemarau"
    return "Peralihan II"


def tafsir_v(v):
    return "sangat kuat" if v >= .5 else "kuat" if v >= .3 else "sedang" if v >= .1 else "lemah"


def epsilon_squared(H, n, k):
    return (H - k + 1) / (n - k)


def tafsir_eff(e):
    return "besar" if e >= .26 else "sedang" if e >= .13 else "kecil" if e >= .04 else "sangat kecil"


def dunn(df, kolom, grup_kol, alpha=ALPHA):
    d = df[[kolom, grup_kol]].dropna().copy()
    n = len(d)
    d["rank"] = stats.rankdata(d[kolom])
    gs = sorted(d[grup_kol].unique())
    mr, ng = d.groupby(grup_kol)["rank"].mean(), d.groupby(grup_kol).size()
    _, cnt = np.unique(d[kolom].values, return_counts=True)
    tie = np.sum(cnt ** 3 - cnt) / (12.0 * (n - 1))
    npair = len(gs) * (len(gs) - 1) / 2
    out = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            a, b = gs[i], gs[j]
            se = np.sqrt((n * (n + 1) / 12.0 - tie) * (1.0 / ng[a] + 1.0 / ng[b]))
            z = (mr[a] - mr[b]) / se
            p = min(2 * (1 - stats.norm.cdf(abs(z))) * npair, 1.0)
            out.append({"variabel": kolom, "pasangan": f"K{a} vs K{b}", "z": round(float(z), 2),
                        "p_bonferroni": f"{p:.2e}", "signifikan": "Ya" if p < alpha else "Tidak"})
    return pd.DataFrame(out)


def buat_model(nama, k, rs, sisi):
    """Model untuk uji stabilitas bootstrap (identik notebook)."""
    if nama == "K-Means":
        return lambda X: KMeans(n_clusters=k, random_state=rs, n_init=10).fit_predict(X)
    if nama == "GMM":
        return lambda X: GaussianMixture(n_components=k, covariance_type="full",
                                         random_state=rs, n_init=3).fit(X).predict(X)
    if nama == "Hierarchical":
        return lambda X: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

    def _som(X):
        s = SOM(sisi, sisi, X.shape[1], n_iter=6000, random_state=rs).fit(X)
        b = s.bmu(X)
        u = np.unique(b)
        lab = KMeans(n_clusters=k, random_state=rs, n_init=10).fit_predict(s.W[u])
        peta = dict(zip(u, lab))
        return np.array([peta[i] for i in b])
    return _som


# ---------------------------------------------------------------------------
# PIPELINE UTAMA
# ---------------------------------------------------------------------------
def run_pipeline(df_raw_input):
    """Menjalankan seluruh alur penelitian, urutan sama dengan notebook.
    Mengembalikan dict berisi semua tabel/array hasil."""
    np.random.seed(RANDOM_STATE)
    R = {}

    df_raw = standardize_raw(df_raw_input)
    feature_cols = [c for c in df_raw.columns if c != "date"]
    R["df_raw"], R["feature_cols"] = df_raw, feature_cols

    # --- Verifikasi ketersediaan parameter (-999) ---
    R["cek_parameter"] = pd.DataFrame({
        "variabel": feature_cols,
        "jumlah_-999": [(df_raw[c] == -999).sum() for c in feature_cols],
        "nilai_min": [df_raw[c].min() for c in feature_cols],
        "nilai_maks": [df_raw[c].max() for c in feature_cols],
    })

    # --- PREPROCESSING ---
    df_clean = df_raw.copy()
    n_before = len(df_clean)
    audit = {}

    df_clean[feature_cols] = df_clean[feature_cols].replace([-999, -999.0], np.nan)
    audit["missing_terdeteksi"] = int(df_clean[feature_cols].isna().sum().sum())

    n_dup = int(df_clean["date"].duplicated().sum())
    audit["tanggal_duplikat"] = n_dup
    if n_dup:
        df_clean = df_clean.drop_duplicates(subset="date", keep="first")

    rentang = pd.date_range(df_clean["date"].min(), df_clean["date"].max(), freq="D")
    hilang = rentang.difference(pd.DatetimeIndex(df_clean["date"]))
    audit["tanggal_tidak_ada"] = len(hilang)
    if len(hilang):
        df_clean = (df_clean.set_index("date").reindex(rentang)
                    .rename_axis("date").reset_index())

    for c in feature_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    valid_ranges = {"suhu_c": (-10, 55), "kelembapan_persen": (0, 100),
                    "curah_hujan_mm": (0, None), "angin_ms": (0, None),
                    "tekanan_kpa": (50, 110), "radiasi_mj_m2": (0, None)}
    invalid_report = {}
    for c, (lo, hi) in valid_ranges.items():
        if c in df_clean.columns:
            m = pd.Series(False, index=df_clean.index)
            if lo is not None:
                m |= df_clean[c] < lo
            if hi is not None:
                m |= df_clean[c] > hi
            invalid_report[c] = int(m.sum())
            df_clean.loc[m, c] = np.nan
    audit["nilai_di_luar_rentang_fisik"] = int(sum(invalid_report.values()))

    df_clean = df_clean.set_index("date")
    audit["missing_sebelum_imputasi"] = int(df_clean[feature_cols].isna().sum().sum())
    df_clean[feature_cols] = df_clean[feature_cols].interpolate(method="time", limit_direction="both")
    audit["missing_sesudah_imputasi"] = int(df_clean[feature_cols].isna().sum().sum())
    df_clean = df_clean.reset_index()

    audit["baris_sebelum"] = n_before
    audit["baris_sesudah"] = len(df_clean)
    audit["kelengkapan_persen"] = round(
        100 * (1 - audit["missing_sebelum_imputasi"] / (len(df_clean) * len(feature_cols))), 4)

    R["audit"] = audit
    R["audit_df"] = pd.DataFrame({"indikator": list(audit), "nilai": list(audit.values())})
    R["invalid_report"] = invalid_report
    R["df_clean"] = df_clean

    # --- DATA EKSPLORATIF ---
    desc = df_clean[feature_cols].describe().T
    desc["median"] = df_clean[feature_cols].median()
    desc["skewness"] = df_clean[feature_cols].skew()
    desc["kurtosis"] = df_clean[feature_cols].kurtosis()
    desc["cv_persen"] = (desc["std"] / desc["mean"] * 100)
    desc = desc.round(3)
    R["desc"] = desc
    R["menceng"] = desc.index[desc["skewness"].abs() > SKEW_THRESHOLD].tolist()

    bln = df_clean["date"].dt.month
    R["profil_bulanan"] = df_clean.groupby(bln)[feature_cols].mean().round(2)
    R["profil_bulanan"].index.name = "bulan"

    # --- KORELASI & REDUNDANSI ---
    corr = df_clean[feature_cols].corr()
    pasangan = [(feature_cols[i], feature_cols[j], float(corr.iloc[i, j]))
                for i in range(len(feature_cols)) for j in range(i + 1, len(feature_cols))
                if abs(corr.iloc[i, j]) >= CORR_WATCH]
    dibuang = set()
    for a, b, r in pasangan:
        if abs(r) >= CORR_DROP and b not in dibuang:
            dibuang.add(b)
    features_final = [f for f in feature_cols if f not in dibuang]
    R.update(corr=corr, pasangan_korelasi=pasangan, dibuang=sorted(dibuang),
             features_final=features_final)

    # --- TRANSFORMASI & STANDARISASI ---
    kandidat = [f for f in features_final
                if abs(df_clean[f].skew()) > SKEW_THRESHOLD and df_clean[f].min() >= 0]
    df_trans = df_clean[features_final].copy()
    log_trans = []
    for f in kandidat:
        df_trans[f] = np.log1p(df_trans[f])
        log_trans.append({"variabel": f,
                          "skew_sebelum": round(float(df_clean[f].skew()), 3),
                          "skew_sesudah": round(float(df_trans[f].skew()), 3),
                          "transformasi": "log1p"})
    R.update(kandidat=kandidat, df_trans=df_trans, log_trans=pd.DataFrame(log_trans))

    sc_A = StandardScaler()
    XA = sc_A.fit_transform(df_clean[features_final])
    sc_B = StandardScaler()
    XB = sc_B.fit_transform(df_trans[features_final])
    df_A = pd.DataFrame(XA, columns=features_final)
    df_B = pd.DataFrame(XB, columns=features_final)
    R["rentang_standar"] = pd.DataFrame({
        "min_A": df_A.min().round(2), "maks_A": df_A.max().round(2),
        "min_B": df_B.min().round(2), "maks_B": df_B.max().round(2)})

    # --- PENENTUAN JUMLAH KLASTER ---
    met_A, met_B = evaluasi_k(XA), evaluasi_k(XB)
    banding = pd.DataFrame({
        "K": met_A["K"],
        "Sil_A_tanpa_transformasi": met_A["Silhouette"].round(4),
        "Sil_B_dengan_log1p": met_B["Silhouette"].round(4),
        "DB_A": met_A["Davies_Bouldin"].round(4),
        "DB_B": met_B["Davies_Bouldin"].round(4),
    })
    kA = int(met_A.loc[met_A["Silhouette"].idxmax(), "K"])
    kB = int(met_B.loc[met_B["Silhouette"].idxmax(), "K"])
    selisih = met_B["Silhouette"].max() - met_A["Silhouette"].max()
    if selisih >= -0.01:
        SKENARIO, X_final, metrics_df = "B (log1p)", XB, met_B
    else:
        SKENARIO, X_final, metrics_df = "A (tanpa transformasi)", XA, met_A
    R.update(banding_skenario=banding, sil_best_A=float(met_A["Silhouette"].max()), kA=kA,
             sil_best_B=float(met_B["Silhouette"].max()), kB=kB, SKENARIO=SKENARIO,
             X_final=X_final)

    k_elbow = cari_elbow(metrics_df["K"].tolist(), metrics_df["Inertia"].tolist())
    k_sil = int(metrics_df.loc[metrics_df["Silhouette"].idxmax(), "K"])
    bic, aic = [], []
    for k in range(K_MIN, K_MAX + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=RANDOM_STATE, n_init=5).fit(X_final)
        bic.append(g.bic(X_final))
        aic.append(g.aic(X_final))
    metrics_df["BIC_GMM"] = bic
    metrics_df["AIC_GMM"] = aic
    k_bic = int(metrics_df.loc[metrics_df["BIC_GMM"].idxmin(), "K"])

    metrics_df["Composite_Score"] = (minmax(metrics_df["Silhouette"])
                                     + minmax(metrics_df["Calinski_Harabasz"])
                                     + (1 - minmax(metrics_df["Davies_Bouldin"]))) / 3
    best_k = int(metrics_df.loc[metrics_df["Composite_Score"].idxmax(), "K"])
    R.update(metrics_df=metrics_df, k_elbow=k_elbow, k_sil=k_sil, k_bic=k_bic, best_k=best_k)

    # --- ALGORITMA KLASTERISASI ---
    n_node_target = 5 * np.sqrt(len(X_final))
    sisi = int(np.ceil(np.sqrt(n_node_target)))
    som = SOM(sisi, sisi, X_final.shape[1], n_iter=12000, random_state=RANDOM_STATE).fit(X_final)
    bmu_idx = som.bmu(X_final)
    node_terpakai = np.unique(bmu_idx)
    km_node = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    label_node_terpakai = km_node.fit_predict(som.W[node_terpakai])
    peta_node = dict(zip(node_terpakai, label_node_terpakai))
    label_som = np.array([peta_node[b] for b in bmu_idx])
    R["som_info"] = {"sisi": sisi, "n_node": sisi ** 2, "target": float(n_node_target),
                     "quantization_error": som.quantization_error(X_final),
                     "node_terpakai": int(len(node_terpakai))}

    hasil_label = {}
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    hasil_label["K-Means"] = km.fit_predict(X_final)
    gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                          random_state=RANDOM_STATE, n_init=10).fit(X_final)
    hasil_label["GMM"] = gmm.predict(X_final)
    gmm_proba = gmm.predict_proba(X_final)
    hasil_label["SOM"] = label_som
    hasil_label["Hierarchical"] = AgglomerativeClustering(
        n_clusters=best_k, linkage="ward").fit_predict(X_final)
    R["hasil_label"] = hasil_label
    R["sebaran_klaster"] = pd.DataFrame(
        {nama: np.bincount(lb, minlength=best_k) for nama, lb in hasil_label.items()}
    ).T.rename(columns=lambda c: f"Klaster {c}")

    # Ambiguitas GMM
    proba_maks = gmm_proba.max(axis=1)
    n_ambigu = int((proba_maks < AMBANG_AMBIGU).sum())
    tmp = pd.DataFrame({"bulan": df_clean["date"].dt.month, "ambigu": proba_maks < AMBANG_AMBIGU})
    R.update(proba_maks=proba_maks, n_ambigu=n_ambigu,
             ambigu_per_bulan=tmp.groupby("bulan")["ambigu"].mean() * 100)

    # Dendrogram (data linkage Ward)
    R["linkage_Z"] = linkage(X_final, method="ward")

    # --- PERBANDINGAN KINERJA ---
    baris = []
    for nama, lb in hasil_label.items():
        baris.append({
            "Algoritma": nama, "K": best_k,
            "Silhouette": silhouette_score(X_final, lb),
            "Calinski_Harabasz": calinski_harabasz_score(X_final, lb),
            "Davies_Bouldin": davies_bouldin_score(X_final, lb),
            "Klaster_terkecil_persen": np.bincount(lb, minlength=best_k).min() / len(lb) * 100,
        })
    perbandingan = pd.DataFrame(baris).round(4)
    R["perbandingan"] = perbandingan

    nama_alg = list(hasil_label)
    ari_mat = pd.DataFrame(np.eye(len(nama_alg)), index=nama_alg, columns=nama_alg)
    nmi_mat = ari_mat.copy()
    for a, b in itertools.combinations(nama_alg, 2):
        ari_mat.loc[a, b] = ari_mat.loc[b, a] = adjusted_rand_score(hasil_label[a], hasil_label[b])
        nmi_mat.loc[a, b] = nmi_mat.loc[b, a] = normalized_mutual_info_score(hasil_label[a], hasil_label[b])
    nilai_luar = [ari_mat.loc[a, b] for a, b in itertools.combinations(nama_alg, 2)]
    R.update(ari_mat=ari_mat, nmi_mat=nmi_mat, ari_antar_mean=float(np.mean(nilai_luar)),
             ari_antar_min=float(np.min(nilai_luar)), ari_antar_max=float(np.max(nilai_luar)))

    R["sil_samples"] = {n: silhouette_samples(X_final, lb) for n, lb in hasil_label.items()}
    R["sil_score"] = {n: silhouette_score(X_final, lb) for n, lb in hasil_label.items()}

    # --- VALIDASI EKSTERNAL TERHADAP MUSIM ---
    musim = df_clean["date"].dt.month.map(label_musim)
    musim_kode = pd.Categorical(musim, categories=URUT_MUSIM).codes

    def validasi_musim(lb):
        kont = pd.crosstab(musim, pd.Series(lb, index=musim.index)).reindex(URUT_MUSIM)
        chi2, p, dof, _ = stats.chi2_contingency(kont.values)
        n = kont.values.sum()
        v = np.sqrt((chi2 / n) / (min(kont.shape[0] - 1, kont.shape[1] - 1)))
        return {"chi2": chi2, "p": p, "dof": dof, "cramers_v": v,
                "ari": adjusted_rand_score(musim_kode, lb),
                "nmi": normalized_mutual_info_score(musim_kode, lb),
                "kontingensi": kont}

    hasil_validasi = {nama: validasi_musim(lb) for nama, lb in hasil_label.items()}
    tabel_validasi = pd.DataFrame([{
        "Algoritma": nama, "Chi_square": round(v["chi2"], 2), "df": v["dof"],
        "p_value": f"{v['p']:.2e}", "Cramers_V": round(v["cramers_v"], 4),
        "ARI_vs_musim": round(v["ari"], 4), "NMI_vs_musim": round(v["nmi"], 4),
    } for nama, v in hasil_validasi.items()])
    R.update(musim=musim, hasil_validasi=hasil_validasi, tabel_validasi=tabel_validasi,
             alg_terbaik_musim=tabel_validasi.loc[tabel_validasi["Cramers_V"].idxmax(), "Algoritma"],
             v_maks=float(tabel_validasi["Cramers_V"].max()))

    # --- PEMILIHAN ALGORITMA TERBAIK ---
    skor = pd.DataFrame({"Algoritma": perbandingan["Algoritma"]})
    skor["sil_norm"] = minmax(perbandingan["Silhouette"])
    skor["db_norm"] = 1 - minmax(perbandingan["Davies_Bouldin"])
    skor["ch_norm"] = minmax(perbandingan["Calinski_Harabasz"])
    skor["v_norm"] = minmax(tabel_validasi["Cramers_V"])
    skor["skor_akhir"] = (BOBOT_INTERNAL * (skor["sil_norm"] + skor["db_norm"] + skor["ch_norm"]) / 3
                          + BOBOT_EKSTERNAL * skor["v_norm"])
    ALG_TERBAIK = skor.loc[skor["skor_akhir"].idxmax(), "Algoritma"]
    labels_final = hasil_label[ALG_TERBAIK]
    R.update(skor=skor, ALG_TERBAIK=ALG_TERBAIK, labels_final=labels_final)

    df_clustered = df_clean.copy()
    df_clustered["cluster"] = labels_final
    df_clustered["musim"] = musim.values
    for nama, lb in hasil_label.items():
        df_clustered[f"cluster_{nama.replace('-', '').lower()}"] = lb

    # --- UJI STABILITAS BOOTSTRAP ---
    rng = np.random.RandomState(RANDOM_STATE)
    rekaman = []
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(X_final), int(SUBSAMPLE_FRAC * len(X_final)), replace=False)
        Xs = X_final[idx]
        lb = buat_model(ALG_TERBAIK, best_k, int(rng.randint(0, 9999)), sisi)(Xs)
        rekaman.append({"iterasi": b + 1,
                        "ari_vs_model_penuh": adjusted_rand_score(labels_final[idx], lb),
                        "silhouette": silhouette_score(Xs, lb)})
    stab = pd.DataFrame(rekaman)
    ari_mean, ari_std = stab["ari_vs_model_penuh"].mean(), stab["ari_vs_model_penuh"].std()
    tafsir_stab = ("struktur klaster sangat stabil terhadap variasi subset data" if ari_mean > .75
                   else "struktur klaster cukup stabil terhadap variasi subset data" if ari_mean > .5
                   else "struktur klaster kurang stabil dan perlu ditafsirkan hati-hati")
    R.update(stab=stab, ari_mean=float(ari_mean), ari_std=float(ari_std), tafsir_stab=tafsir_stab,
             sil_final=float(silhouette_score(X_final, labels_final)))

    # --- KARAKTERISTIK & PENAMAAN KLASTER ---
    cluster_mean = df_clustered.groupby("cluster")[features_final].mean().round(3)
    cluster_std = df_clustered.groupby("cluster")[features_final].std().round(3)
    cluster_counts = df_clustered["cluster"].value_counts().sort_index()
    cluster_pct = (cluster_counts / len(df_clustered) * 100).round(2)
    gmean, gstd = df_clustered[features_final].mean(), df_clustered[features_final].std()
    z_table = ((cluster_mean - gmean) / gstd).round(2)

    interpretasi = {}
    for c in z_table.index:
        bagian = []
        for f in features_final:
            z = z_table.loc[c, f]
            tk = ("relatif tinggi" if z > Z_THRESHOLD else "relatif rendah" if z < -Z_THRESHOLD
                  else "mendekati rata-rata")
            bagian.append(f"{f}: {tk} (z={z:+.2f})")
        interpretasi[c] = bagian

    # Penamaan manual peneliti hanya berlaku bila hasil run ini identik dengan
    # sebaran klaster pada penelitian (K=3, K-Means, jumlah anggota sama).
    nama_berlaku = (best_k == len(CLUSTER_NAMES_PUBLISHED) and ALG_TERBAIK == "K-Means"
                    and cluster_counts.tolist() == CLUSTER_SIZES_PUBLISHED)
    cluster_names = dict(CLUSTER_NAMES_PUBLISHED) if nama_berlaku else {}
    for c in sorted(df_clustered["cluster"].unique()):
        cluster_names.setdefault(c, f"Klaster {c}")
    df_clustered["nama_klaster"] = df_clustered["cluster"].map(cluster_names)

    R.update(cluster_mean=cluster_mean, cluster_std=cluster_std, cluster_counts=cluster_counts,
             cluster_pct=cluster_pct, z_table=z_table, interpretasi=interpretasi,
             cluster_names=cluster_names, nama_berlaku=nama_berlaku)

    # --- POLA TEMPORAL ---
    df_clustered["year"] = df_clustered["date"].dt.year
    df_clustered["month"] = df_clustered["date"].dt.month
    bulanan = df_clustered.groupby(["month", "cluster"]).size().unstack(fill_value=0)
    bulanan.index = BULAN
    bulanan_pct = bulanan.div(bulanan.sum(axis=1), axis=0) * 100
    tahunan = df_clustered.groupby(["year", "cluster"]).size().unstack(fill_value=0)
    tahunan_pct = (tahunan.div(tahunan.sum(axis=1), axis=0) * 100).round(2)
    R.update(bulanan=bulanan, bulanan_pct=bulanan_pct, tahunan=tahunan, tahunan_pct=tahunan_pct)

    # --- VISUALISASI HASIL (centroid, PCA) ---
    centroid = pd.DataFrame(
        np.vstack([X_final[labels_final == c].mean(axis=0) for c in range(best_k)]),
        columns=features_final)
    pca = PCA(n_components=2)
    P = pca.fit_transform(X_final)
    ev = pca.explained_variance_ratio_ * 100
    loading = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features_final).round(3)
    R.update(centroid=centroid, P=P, ev=ev, loading=loading)

    # --- PERBEDAAN ANTAR KLASTER ---
    grup = {c: df_clustered[df_clustered["cluster"] == c] for c in sorted(df_clustered["cluster"].unique())}
    n_obs = len(df_clustered)
    hasil_uji = []
    for f in features_final:
        g = [grup[c][f].values for c in sorted(grup)]
        normal = all(stats.normaltest(x)[1] > ALPHA for x in g if len(x) >= 8)
        homogen = stats.levene(*g)[1] > ALPHA
        if normal and homogen:
            s, p = stats.f_oneway(*g)
            uji = "ANOVA"
            allv = np.concatenate(g)
            eff = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in g) / ((allv - allv.mean()) ** 2).sum()
            jenis = "eta_squared"
        else:
            s, p = stats.kruskal(*g)
            uji = "Kruskal-Wallis"
            eff = epsilon_squared(s, n_obs, best_k)
            jenis = "epsilon_squared"
        hasil_uji.append({"variabel": f, "uji": uji, "normalitas": normal, "homogenitas": homogen,
                          "statistik": round(float(s), 2), "p_value": f"{p:.2e}",
                          "effect_size": round(float(eff), 4), "jenis": jenis,
                          "kekuatan": tafsir_eff(float(eff))})
    uji_df = pd.DataFrame(hasil_uji)

    posthoc = pd.concat([dunn(df_clustered, f, "cluster") for f in features_final], ignore_index=True)
    ringkas = posthoc.groupby("pasangan")["signifikan"].apply(
        lambda s: f"{(s == 'Ya').sum()}/{len(s)} variabel berbeda signifikan")
    pasangan_lemah = ringkas[ringkas.str.startswith(("0/", "1/", "2/"))]
    R.update(uji_df=uji_df, posthoc=posthoc, ringkas_posthoc=ringkas, pasangan_lemah=pasangan_lemah)

    R["df_clustered"] = df_clustered
    return R


# ---------------------------------------------------------------------------
# NILAI REFERENSI DARI OUTPUT NOTEBOOK PENELITIAN (untuk cek reproduksi)
# Disalin dari output sel notebook; bukan hasil perhitungan aplikasi.
# ---------------------------------------------------------------------------
REFERENSI_NOTEBOOK = {
    "Jumlah observasi harian": 4383,
    "Fitur final": 6,
    "Skenario terpilih": "B (log1p)",
    "K Elbow": 4,
    "K Silhouette": 3,
    "K BIC (GMM)": 8,
    "K digunakan (skor gabungan)": 3,
    "Silhouette K-Means": 0.2748,
    "Silhouette GMM": 0.1708,
    "Silhouette SOM": 0.2576,
    "Silhouette Hierarchical": 0.2231,
    "Cramér's V K-Means": 0.6321,
    "Cramér's V GMM": 0.5721,
    "Cramér's V SOM": 0.6270,
    "Cramér's V Hierarchical": 0.6324,
    "Algoritma terpilih": "K-Means",
    "Sebaran klaster K-Means": [1379, 1990, 1014],
    "Rata-rata ARI antaralgoritma": 0.5543,
    "Stabilitas bootstrap (rata-rata ARI)": 0.9800,
    "Hari ambigu GMM": 114,
    "Varians PC1 (%)": 43.14,
    "Varians PC2 (%)": 29.12,
}


def cek_reproduksi(R):
    """Membandingkan hasil run ini dengan output notebook penelitian."""
    tv = R["tabel_validasi"].set_index("Algoritma")["Cramers_V"]
    pb = R["perbandingan"].set_index("Algoritma")["Silhouette"]
    hasil = {
        "Jumlah observasi harian": len(R["df_clustered"]),
        "Fitur final": len(R["features_final"]),
        "Skenario terpilih": R["SKENARIO"],
        "K Elbow": R["k_elbow"], "K Silhouette": R["k_sil"], "K BIC (GMM)": R["k_bic"],
        "K digunakan (skor gabungan)": R["best_k"],
        "Silhouette K-Means": pb["K-Means"], "Silhouette GMM": pb["GMM"],
        "Silhouette SOM": pb["SOM"], "Silhouette Hierarchical": pb["Hierarchical"],
        "Cramér's V K-Means": tv["K-Means"], "Cramér's V GMM": tv["GMM"],
        "Cramér's V SOM": tv["SOM"], "Cramér's V Hierarchical": tv["Hierarchical"],
        "Algoritma terpilih": R["ALG_TERBAIK"],
        "Sebaran klaster K-Means": np.bincount(R["hasil_label"]["K-Means"], minlength=R["best_k"]).tolist(),
        "Rata-rata ARI antaralgoritma": round(R["ari_antar_mean"], 4),
        "Stabilitas bootstrap (rata-rata ARI)": round(R["ari_mean"], 4),
        "Hari ambigu GMM": R["n_ambigu"],
        "Varians PC1 (%)": round(float(R["ev"][0]), 2),
        "Varians PC2 (%)": round(float(R["ev"][1]), 2),
    }
    rows = []
    for k, ref in REFERENSI_NOTEBOOK.items():
        nilai = hasil[k]
        if isinstance(ref, float):
            ok = abs(float(nilai) - ref) < 1e-4 + 1e-9
        else:
            ok = nilai == ref
        rows.append({"Item": k, "Notebook (publikasi)": str(ref), "Aplikasi (run ini)": str(nilai),
                     "Sesuai": "✅" if ok else "❌"})
    return pd.DataFrame(rows)