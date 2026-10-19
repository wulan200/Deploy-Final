"""
plots.py
========
Seluruh gambar pada notebook penelitian, dibungkus sebagai fungsi yang
mengembalikan objek Figure (tanpa plt.show / savefig) agar dapat ditampilkan
dengan st.pyplot.
"""
from math import pi

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif untuk server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from pipeline import AMBANG_AMBIGU

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 110


def _grid(feature_cols, ncol=3):
    nrow = int(np.ceil(len(feature_cols) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(15, 4 * nrow))
    ax = np.atleast_1d(ax).flatten()
    for j in range(len(feature_cols), len(ax)):
        fig.delaxes(ax[j])
    return fig, ax


# ---------------------------------------------------------------- Eksploratif
def fig_histogram(df_clean, feature_cols):
    fig, ax = _grid(feature_cols)
    for i, c in enumerate(feature_cols):
        ax[i].hist(df_clean[c], bins=45, color="steelblue", edgecolor="white")
        ax[i].set_title(f"{c} (skew={df_clean[c].skew():.2f})")
    fig.tight_layout()
    return fig


def fig_boxplot(df_clean, feature_cols):
    fig, ax = _grid(feature_cols)
    for i, c in enumerate(feature_cols):
        ax[i].boxplot(df_clean[c].dropna())
        ax[i].set_title(c)
    fig.tight_layout()
    return fig


def fig_profil_bulanan(profil_bulanan, feature_cols):
    fig, ax = _grid(feature_cols)
    for i, c in enumerate(feature_cols):
        ax[i].plot(profil_bulanan.index, profil_bulanan[c], marker="o", color="darkred")
        ax[i].set_title(f"Rata-rata bulanan: {c}")
        ax[i].set_xticks(range(1, 13))
    fig.tight_layout()
    return fig


def fig_korelasi(corr):
    fig = plt.figure(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=.5)
    plt.title("Matriks Korelasi Antarvariabel Meteorologi")
    fig.tight_layout()
    return fig


def fig_transformasi(df_clean, df_trans, kandidat):
    if not kandidat:
        return None
    fig, ax = plt.subplots(len(kandidat), 2, figsize=(11, 3.4 * len(kandidat)))
    ax = np.atleast_2d(ax)
    for i, f in enumerate(kandidat):
        ax[i, 0].hist(df_clean[f], bins=45, color="indianred", edgecolor="white")
        ax[i, 0].set_title(f"{f} asli (skew={df_clean[f].skew():.2f})")
        ax[i, 1].hist(df_trans[f], bins=45, color="seagreen", edgecolor="white")
        ax[i, 1].set_title(f"{f} log1p (skew={df_trans[f].skew():.2f})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Penentuan K
def fig_penentuan_k(metrics_df, k_elbow, k_sil, k_bic):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(metrics_df["K"], metrics_df["Inertia"], marker="o")
    ax[0].axvline(k_elbow, color="red", ls="--", label=f"Elbow K={k_elbow}")
    ax[0].set_title("Elbow Method"); ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia"); ax[0].legend()

    ax[1].plot(metrics_df["K"], metrics_df["Silhouette"], marker="o", color="green")
    ax[1].axvline(k_sil, color="red", ls="--", label=f"Maks K={k_sil}")
    ax[1].set_title("Silhouette vs K"); ax[1].set_xlabel("K"); ax[1].legend()

    ax[2].plot(metrics_df["K"], metrics_df["BIC_GMM"], marker="o", label="BIC")
    ax[2].plot(metrics_df["K"], metrics_df["AIC_GMM"], marker="s", label="AIC", alpha=.7)
    ax[2].axvline(k_bic, color="red", ls="--", label=f"BIC min K={k_bic}")
    ax[2].set_title("BIC/AIC GMM"); ax[2].set_xlabel("K"); ax[2].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Klasterisasi
def fig_ambiguitas(proba_maks, pers):
    fig = plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.hist(proba_maks, bins=40, color="mediumpurple", edgecolor="white")
    plt.axvline(AMBANG_AMBIGU, color="red", ls="--", label=f"Ambang {AMBANG_AMBIGU}")
    plt.xlabel("Probabilitas keanggotaan tertinggi"); plt.ylabel("Jumlah hari")
    plt.title("Ketegasan Keanggotaan (GMM)"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(pers.index, pers.values, color="coral")
    plt.xlabel("Bulan"); plt.ylabel("% hari ambigu"); plt.xticks(range(1, 13))
    plt.title("Sebaran Hari Peralihan per Bulan")
    fig.tight_layout()
    return fig


def fig_dendrogram(Z):
    fig = plt.figure(figsize=(11, 4.6))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True,
               leaf_rotation=90, leaf_font_size=9)
    plt.title("Dendrogram Hierarchical Clustering (Ward, dipangkas 30 node)")
    plt.xlabel("Indeks sampel atau (jumlah anggota node)"); plt.ylabel("Jarak Ward")
    fig.tight_layout()
    return fig


def fig_pca(P, ev, hasil_label, sil_score, best_k):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for a, (nama, lb) in zip(axes.flatten(), hasil_label.items()):
        for c in range(best_k):
            m = lb == c
            a.scatter(P[m, 0], P[m, 1], s=7, alpha=.5, label=f"K{c}")
        a.set_title(f"{nama} (Sil={sil_score[nama]:.3f})")
        a.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); a.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
        a.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    return fig


def fig_distribusi_centroid(cluster_counts, cluster_pct, centroid, alg, best_k):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ax[0].bar([f"K{i}" for i in cluster_counts.index], cluster_counts.values,
              color=sns.color_palette("tab10", best_k))
    for i, v in enumerate(cluster_counts.values):
        ax[0].text(i, v, f"{v}\n({cluster_pct.values[i]:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax[0].set_ylabel("Jumlah hari"); ax[0].set_title(f"Distribusi Anggota Klaster ({alg})")
    sns.heatmap(centroid, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1],
                yticklabels=[f"K{i}" for i in range(best_k)], linewidths=.5)
    ax[1].set_title("Centroid (skala terstandardisasi)")
    fig.tight_layout()
    return fig


def fig_radar(centroid, cluster_names, features_final, best_k):
    if not (best_k <= 6 and len(features_final) <= 8):
        return None
    N = len(features_final)
    sudut = [n / float(N) * 2 * pi for n in range(N)] + [0]
    fig = plt.figure(figsize=(7.5, 7.5))
    a = plt.subplot(111, polar=True)
    for c in range(best_k):
        v = centroid.loc[c].tolist()
        v += v[:1]
        a.plot(sudut, v, lw=1.8, label=f"K{c}: {cluster_names[c]}")
        a.fill(sudut, v, alpha=.08)
    a.set_xticks(sudut[:-1]); a.set_xticklabels(features_final, fontsize=9)
    plt.title("Radar Chart Karakteristik Klaster", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.45, 1.12), fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Evaluasi
def fig_perbandingan_metrik(perbandingan):
    """Tiga panel metrik internal. Batang hijau = nilai terbaik.

    CATATAN PERUBAHAN: pada notebook, kondisi warna
    `(v == max) == tinggi_baik` membuat panel Davies-Bouldin menandai SEMUA batang
    selain yang tertinggi sebagai hijau. Di sini, batang hijau menandai nilai
    TERBAIK (maks untuk Silhouette/CH, min untuk Davies-Bouldin). Angka tidak berubah.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, (m, judul, tinggi_baik) in enumerate([
            ("Silhouette", "Silhouette (makin tinggi makin baik)", True),
            ("Calinski_Harabasz", "Calinski-Harabasz (makin tinggi makin baik)", True),
            ("Davies_Bouldin", "Davies-Bouldin (makin rendah makin baik)", False)]):
        terbaik = perbandingan[m].max() if tinggi_baik else perbandingan[m].min()
        warna = ["seagreen" if v == terbaik else "steelblue" for v in perbandingan[m]]
        ax[i].bar(perbandingan["Algoritma"], perbandingan[m], color=warna)
        ax[i].set_title(judul, fontsize=9)
        ax[i].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def fig_kesepakatan(ari_mat, nmi_mat):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    sns.heatmap(ari_mat.astype(float), annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0, vmax=1, ax=ax[0], linewidths=.5)
    ax[0].set_title("Adjusted Rand Index Antaralgoritma")
    sns.heatmap(nmi_mat.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax[1], linewidths=.5)
    ax[1].set_title("Normalized Mutual Information Antaralgoritma")
    fig.tight_layout()
    return fig


def fig_silhouette(hasil_label, sil_samples, sil_score, best_k):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    palet = sns.color_palette("tab10", best_k)
    for idx, (nama, lb) in enumerate(hasil_label.items()):
        sv, rata = sil_samples[nama], sil_score[nama]
        a = axes[idx]
        y = 10
        for c in range(best_k):
            v = np.sort(sv[lb == c])
            a.fill_betweenx(np.arange(y, y + len(v)), 0, v,
                            facecolor=palet[c], edgecolor=palet[c], alpha=.8)
            a.text(-0.08, y + .5 * len(v), str(c), fontsize=8)
            y += len(v) + 10
        a.axvline(rata, color="red", ls="--", label=f"Rata-rata={rata:.3f}")
        a.set_title(f"{nama} (negatif: {(sv < 0).sum() / len(sv) * 100:.1f}%)")
        a.set_yticks([]); a.set_xlabel("Koefisien silhouette"); a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_stabilitas(stab, ari_mean, sil_final):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(stab["ari_vs_model_penuh"], bins=15, color="steelblue", edgecolor="white")
    ax[0].axvline(ari_mean, color="red", ls="--", label=f"Rata-rata={ari_mean:.3f}")
    ax[0].set_xlabel("Adjusted Rand Index"); ax[0].set_ylabel("Frekuensi")
    ax[0].set_title("Distribusi ARI Bootstrap"); ax[0].legend()
    ax[1].plot(stab["iterasi"], stab["silhouette"], marker=".")
    ax[1].axhline(sil_final, color="red", ls="--", label="Model penuh")
    ax[1].set_xlabel("Iterasi"); ax[1].set_ylabel("Silhouette")
    ax[1].set_title("Silhouette per Iterasi"); ax[1].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Validasi musim
def fig_validasi_musim(hasil_validasi):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for a, (nama, v) in zip(axes.flatten(), hasil_validasi.items()):
        pers = v["kontingensi"].div(v["kontingensi"].sum(axis=1), axis=0) * 100
        sns.heatmap(pers, annot=True, fmt=".1f", cmap="YlOrRd", ax=a, linewidths=.5,
                    cbar_kws={"label": "% hari"})
        a.set_title(f"{nama} (Cramer's V = {v['cramers_v']:.3f})")
        a.set_xlabel("Klaster"); a.set_ylabel("")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Interpretasi
def fig_zscore(z_table, best_k):
    fig = plt.figure(figsize=(9, .85 * best_k + 2))
    sns.heatmap(z_table, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=.5,
                yticklabels=[f"Klaster {i}" for i in z_table.index])
    plt.title("Profil z-score Tiap Klaster terhadap Rata-rata Keseluruhan")
    fig.tight_layout()
    return fig


def fig_pola_bulanan(bulanan_pct):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    bulanan_pct.plot(kind="bar", stacked=True, colormap="tab10", ax=ax[0], width=.85)
    ax[0].set_ylabel("Persentase (%)"); ax[0].set_xlabel("Bulan")
    ax[0].set_title("Komposisi Klaster per Bulan")
    ax[0].legend(title="Klaster", bbox_to_anchor=(1, 1), fontsize=8)
    sns.heatmap(bulanan_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[1], linewidths=.5)
    ax[1].set_title("Heatmap Persentase Klaster per Bulan"); ax[1].set_xlabel("Klaster")
    fig.tight_layout()
    return fig


def fig_pola_tahunan(tahunan_pct, cluster_names):
    fig = plt.figure(figsize=(11, 5))
    for c in tahunan_pct.columns:
        plt.plot(tahunan_pct.index, tahunan_pct[c], marker="o", label=f"K{c}: {cluster_names[c]}")
    plt.xlabel("Tahun"); plt.ylabel("Persentase hari dalam setahun (%)")
    plt.title("Variasi Antartahun Komposisi Pola Cuaca")
    plt.legend(fontsize=8)
    fig.tight_layout()
    return fig
