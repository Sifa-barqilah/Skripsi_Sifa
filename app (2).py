# -*- coding: utf-8 -*-
"""
Segmentasi Pelanggan K-Means — Queen Buket Majalaya (versi Streamlit)
======================================================================
Alur: Data -> Preprocessing -> Normalisasi -> Elbow & Silhouette -> Clustering
      -> Hasil & Interpretasi Strategi oleh AI (Google Gemini)

Jalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# KONFIGURASI HALAMAN & PALET WARNA
# ============================================================
st.set_page_config(
    page_title="Segmentasi Pelanggan — Queen Buket Majalaya",
    page_icon="🌸",
    layout="wide",
)

INK = "#3B2A32"
INK_SOFT = "#7A6670"
ROSE_DEEP = "#8C3A50"
ROSE = "#B5546B"
SAGE_DEEP = "#57684C"
GOLD = "#C6923C"
CREAM = "#F3EAE0"
DANGER = "#B23B3B"
LINE = "#E7DDD4"

CLUSTER_PALETTE = ["#8C3A50", "#C6923C", "#57684C", "#4A6E8C", "#8C5A3A"]

CUSTOM_CSS = f"""
<style>
.stApp {{ background-color: #FBF7F1; }}
h1, h2, h3 {{ color: {INK}; font-family: 'Georgia', serif; }}
.card {{
    background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 14px;
}}
.eyebrow {{
    font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;
    color: {ROSE}; font-weight: 700; margin-bottom: 4px;
}}
.stButton>button {{
    border-radius: 999px; font-weight: 600; border: 1px solid {ROSE_DEEP};
}}
.stButton>button[kind="primary"] {{ background-color: {ROSE_DEEP}; color: white; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# DATA REFERENSI
# ============================================================
PRODUCTS = [
    ("Buket Bunga", 150000),
    ("Buket Boneka", 210000),
    ("Buket Uang", 260000),
    ("Buket Mercy", 110000),
    ("Buket Round Satin", 185000),
]
PAYMENTS = ["Tunai", "Transfer", "QRIS"]
FIRST_NAMES = ["Siti", "Ahmad", "Dewi", "Rian", "Putri", "Fajar", "Rina", "Doni", "Lestari", "Bayu",
               "Sri", "Agus", "Nita", "Yudi", "Wulan", "Hendra", "Ayu", "Iwan", "Sinta", "Rizky",
               "Dian", "Andri", "Fitri", "Hadi", "Melani", "Wahyu", "Tita", "Eko", "Ratna", "Bagus",
               "Nur", "Dodi", "Vina", "Arif", "Sari", "Adi", "Yuni", "Tono", "Indah", "Rudi"]
LAST_NAMES = ["Permata", "Wijaya", "Kusuma", "Santoso", "Rahayu", "Saputra", "Anggraini", "Setiawan",
              "Utami", "Prasetyo", "Handayani", "Nugraha", "Puspita", "Firmansyah", "Maharani",
              "Gunawan", "Safitri", "Hidayat", "Lestari", "Kurniawan"]

FEATURE_KEYS = ["frekuensi_pembelian", "jumlah_transaksi", "total_nilai_pembelian"]
FEATURE_LABELS = {
    "frekuensi_pembelian": "Frekuensi Pembelian",
    "jumlah_transaksi": "Jumlah Transaksi",
    "total_nilai_pembelian": "Total Nilai Pembelian",
}


def fmt_idr(n):
    return "Rp " + f"{round(n):,}".replace(",", ".")


# ============================================================
# 1. DATA SINTETIS (raw transaksi)
# ============================================================
def generate_synthetic_transactions(seed=None):
    rng = random.Random(seed)
    segments = [
        {"key": "loyal", "weight": 0.22, "freq": (8, 15), "qty": (2, 5)},
        {"key": "potensial", "weight": 0.45, "freq": (3, 7), "qty": (1, 3)},
        {"key": "rendah", "weight": 0.33, "freq": (1, 2), "qty": (1, 2)},
    ]
    n_customers = 86
    customers = []
    for _ in range(n_customers):
        r = rng.random()
        acc = 0.0
        seg = segments[-1]
        for s in segments:
            acc += s["weight"]
            if r <= acc:
                seg = s
                break
        nama = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        telp = f"08{rng.randint(1,9)}{rng.randint(1000,9999)}{rng.randint(1000,9999)}"
        customers.append({"nama": nama, "telp": telp, "seg": seg})

    def rand_date(start, end):
        delta = (end - start).days
        return start + timedelta(days=rng.randint(0, max(delta, 0)))

    start_range = datetime(2025, 6, 1)
    end_range = datetime(2026, 5, 31)

    rows = []
    rid = 1
    for c in customers:
        freq = rng.randint(*c["seg"]["freq"])
        for _ in range(freq):
            produk, harga = rng.choice(PRODUCTS)
            jumlah = rng.randint(*c["seg"]["qty"])
            tgl_pesan = rand_date(start_range, end_range)
            tgl_ambil = tgl_pesan + timedelta(days=rng.randint(0, 4))
            rows.append({
                "id": rid,
                "tanggal_pesan": tgl_pesan.strftime("%Y-%m-%d"),
                "nama_pelanggan": c["nama"],
                "no_telepon": c["telp"],
                "jenis_produk": produk,
                "jumlah_pesanan": jumlah,
                "harga_satuan": harga,
                "metode_pembayaran": rng.choice(PAYMENTS),
                "tanggal_ambil": tgl_ambil.strftime("%Y-%m-%d"),
            })
            rid += 1

    # sisipkan duplikat & nilai kosong agar preprocessing bermakna
    dup_count = round(len(rows) * 0.035)
    for _ in range(dup_count):
        src = dict(rows[rng.randint(0, len(rows) - 1)])
        src["id"] = rid
        rid += 1
        rows.append(src)

    miss_count = round(len(rows) * 0.04)
    for _ in range(miss_count):
        idx = rng.randint(0, len(rows) - 1)
        field = rng.choice(["no_telepon", "jumlah_pesanan", "metode_pembayaran"])
        rows[idx][field] = ""

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = i + 1

    return pd.DataFrame(rows)


# ============================================================
# 2. PREPROCESSING
# ============================================================
def preprocess(raw_df: pd.DataFrame):
    log = []
    df = raw_df.copy()
    before = len(df)

    key_cols = ["tanggal_pesan", "nama_pelanggan", "no_telepon", "jenis_produk", "jumlah_pesanan"]
    df = df.drop_duplicates(subset=key_cols, keep="first")
    dup_removed = before - len(df)
    if dup_removed > 0:
        log.append(f"Menghapus {dup_removed} baris data duplikat.")

    def is_blank(x):
        return pd.isna(x) or str(x).strip() == ""

    mask_valid_name_phone = ~(df["nama_pelanggan"].apply(is_blank) | df["no_telepon"].apply(is_blank))
    missing_dropped = (~mask_valid_name_phone).sum()
    df = df[mask_valid_name_phone].copy()
    if missing_dropped > 0:
        log.append(f"Membuang {missing_dropped} baris karena nama/no. telepon kosong.")

    missing_filled = 0

    def fix_jumlah(v):
        nonlocal missing_filled
        try:
            n = float(v)
            if np.isnan(n) or n <= 0:
                missing_filled += 1
                return 1
            return int(n)
        except Exception:
            missing_filled += 1
            return 1

    def fix_metode(v):
        nonlocal missing_filled
        if is_blank(v):
            missing_filled += 1
            return "Tunai"
        return v

    price_map = dict(PRODUCTS)

    def fix_harga(row):
        try:
            h = float(row["harga_satuan"])
            if np.isnan(h) or h <= 0:
                return price_map.get(row["jenis_produk"], 150000)
            return h
        except Exception:
            return price_map.get(row["jenis_produk"], 150000)

    df["jumlah_pesanan"] = df["jumlah_pesanan"].apply(fix_jumlah)
    df["metode_pembayaran"] = df["metode_pembayaran"].apply(fix_metode)
    df["harga_satuan"] = df.apply(fix_harga, axis=1)
    df["no_telepon"] = df["no_telepon"].astype(str).str.replace(r"\D", "", regex=True)

    if missing_filled > 0:
        log.append(f"Mengisi {missing_filled} nilai kosong (jumlah pesanan / metode pembayaran) dengan nilai default.")
    log.append("Menyesuaikan format tanggal & nomor telepon agar konsisten.")
    log.append(f"Total data bersih: {len(df)} baris (dari {before} baris mentah).")

    return {"data": df, "log": log, "before": before, "after": len(df)}


# ============================================================
# 3. AGREGASI PER PELANGGAN
# ============================================================
def aggregate_by_customer(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()
    df["nilai_transaksi"] = df["jumlah_pesanan"].astype(float) * df["harga_satuan"].astype(float)
    agg = df.groupby("no_telepon").agg(
        nama_pelanggan=("nama_pelanggan", "first"),
        frekuensi_pembelian=("no_telepon", "count"),
        jumlah_transaksi=("jumlah_pesanan", "sum"),
        total_nilai_pembelian=("nilai_transaksi", "sum"),
    ).reset_index()
    return agg


# ============================================================
# 4. NORMALISASI MIN-MAX
# ============================================================
def normalize_minmax(agg_df: pd.DataFrame):
    mins, maxs = {}, {}
    norm = agg_df.copy()
    for k in FEATURE_KEYS:
        mn, mx = agg_df[k].min(), agg_df[k].max()
        mins[k], maxs[k] = mn, mx
        rng = mx - mn
        norm[f"n_{k}"] = 0.0 if rng == 0 else (agg_df[k] - mn) / rng
    return norm, mins, maxs


# ============================================================
# 5. ELBOW (WCSS) + SILHOUETTE SCORE
# ============================================================
def compute_wcss_silhouette(points: np.ndarray, max_k: int, seed=42):
    wcss_series, sil_series = [], []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(points)
        wcss_series.append({"k": k, "wcss": model.inertia_})
        if 2 <= k <= len(points) - 1:
            try:
                sil = silhouette_score(points, labels)
            except Exception:
                sil = np.nan
            sil_series.append({"k": k, "silhouette": sil})
    return wcss_series, sil_series


def detect_elbow(wcss_series):
    """Titik dengan jarak tegak lurus terjauh dari garis (k1,wcss1)-(kN,wcssN)."""
    if len(wcss_series) < 3:
        return wcss_series[0]["k"] if wcss_series else 1
    x1, y1 = wcss_series[0]["k"], wcss_series[0]["wcss"]
    x2, y2 = wcss_series[-1]["k"], wcss_series[-1]["wcss"]
    norm = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) or 1
    best_k, best_dist = x1, -1
    for pt in wcss_series:
        dist = abs((y2 - y1) * pt["k"] - (x2 - x1) * pt["wcss"] + x2 * y1 - y2 * x1) / norm
        if dist > best_dist:
            best_dist, best_k = dist, pt["k"]
    return best_k


def best_silhouette_k(sil_series):
    if not sil_series:
        return None
    valid = [s for s in sil_series if not np.isnan(s["silhouette"])]
    if not valid:
        return None
    return max(valid, key=lambda s: s["silhouette"])["k"]


# ============================================================
# 6. K-MEANS FINAL
# ============================================================
def run_kmeans(points: np.ndarray, k: int, seed=42):
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(points)
    return labels, model.cluster_centers_, model.inertia_


def interpret_clusters(centroids_raw, sizes):
    scores = [
        c["frekuensi_pembelian"] * 0.4 + c["jumlah_transaksi"] * 0.25 + (c["total_nilai_pembelian"] / 1000) * 0.35
        for c in centroids_raw
    ]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    labels = ["Pelanggan Loyal", "Pelanggan Potensial", "Pelanggan Aktivitas Rendah",
              "Segmen Tambahan", "Segmen Tambahan"]
    descs = [
        "Frekuensi & nilai pembelian tinggi — prioritaskan program loyalitas dan penawaran eksklusif.",
        "Aktivitas sedang, berpotensi ditingkatkan lewat promosi & pengingat yang tepat waktu.",
        "Frekuensi & nilai pembelian rendah — perlu pendekatan pemasaran ulang (re-engagement).",
        "Segmen tambahan hasil clustering.",
        "Segmen tambahan hasil clustering.",
    ]
    result = {}
    for rank, idx in enumerate(order):
        result[idx] = {"label": labels[rank] if rank < len(labels) else f"Segmen {rank+1}",
                        "desc": descs[rank] if rank < len(descs) else descs[3], "rank": rank}
    return result


# ============================================================
# STATE INIT
# ============================================================
def reset_pipeline_state():
    for key in ["clean_result", "agg_df", "norm_df", "mins", "maxs",
                "wcss_series", "sil_series", "elbow_k", "sil_k",
                "cluster_result", "gemini_models", "ai_result"]:
        st.session_state.pop(key, None)


if "raw_df" not in st.session_state:
    st.session_state.raw_df = generate_synthetic_transactions(seed=random.randint(0, 999999))
    st.session_state.data_source = "sintetis"

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="eyebrow">Toko Queen Buket Majalaya</div>
    <h1 style="margin-top:0;">🌸 Segmentasi Pelanggan — K-Means</h1>
    <p style="color:{INK_SOFT}; max-width:760px; font-size:15px; line-height:1.6;">
    Alur lengkap: data penjualan → preprocessing → normalisasi Min-Max → evaluasi
    Elbow Method &amp; Silhouette Score → clustering K-Means → interpretasi &amp;
    rekomendasi strategi otomatis oleh model AI Google (Gemini).
    </p>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR — SUMBER DATA
# ============================================================
with st.sidebar:
    st.markdown("### 🌱 Sumber Data")
    st.caption(f"Sumber aktif: **{st.session_state.data_source}** · {len(st.session_state.raw_df)} baris")

    if st.button("🔄 Generate Ulang Data Sintetis", use_container_width=True):
        st.session_state.raw_df = generate_synthetic_transactions(seed=random.randint(0, 999999))
        st.session_state.data_source = "sintetis"
        reset_pipeline_state()
        st.rerun()

    st.markdown("---")
    st.caption("Atau unggah CSV sendiri. Kolom wajib: `tanggal_pesan, nama_pelanggan, no_telepon, "
                "jenis_produk, jumlah_pesanan`. Opsional: `harga_satuan, metode_pembayaran, tanggal_ambil`.")
    uploaded = st.file_uploader("Unggah CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            required = ["tanggal_pesan", "nama_pelanggan", "no_telepon", "jenis_produk", "jumlah_pesanan"]
            missing_cols = [c for c in required if c not in df_up.columns]
            if missing_cols:
                st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
            else:
                if "harga_satuan" not in df_up.columns:
                    price_map = dict(PRODUCTS)
                    df_up["harga_satuan"] = df_up["jenis_produk"].map(price_map).fillna(150000)
                if "metode_pembayaran" not in df_up.columns:
                    df_up["metode_pembayaran"] = "Tunai"
                if "tanggal_ambil" not in df_up.columns:
                    df_up["tanggal_ambil"] = df_up["tanggal_pesan"]
                df_up.insert(0, "id", range(1, len(df_up) + 1))
                if st.session_state.get("_last_upload_name") != uploaded.name:
                    st.session_state.raw_df = df_up
                    st.session_state.data_source = "upload"
                    st.session_state._last_upload_name = uploaded.name
                    reset_pipeline_state()
                    st.rerun()
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

# ============================================================
# TABS
# ============================================================
tab_data, tab_prep, tab_norm, tab_cluster, tab_result = st.tabs(
    ["1️⃣ Data", "2️⃣ Preprocessing", "3️⃣ Normalisasi", "4️⃣ Elbow & Clustering", "5️⃣ Hasil & Interpretasi AI"]
)

raw_df = st.session_state.raw_df

# ---------------- TAB 1: DATA ----------------
with tab_data:
    st.subheader("Data Transaksi Penjualan")
    st.caption("Atribut mengikuti skripsi: tanggal pemesanan, nama pelanggan, no. telepon, jenis produk, "
               "jumlah pemesanan, metode pembayaran, tanggal pengambilan.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Baris Transaksi", len(raw_df))
    c2.metric("Pelanggan Unik (mentah)", raw_df["no_telepon"].nunique())
    c3.metric("Sumber Data", st.session_state.data_source.capitalize())
    st.dataframe(raw_df.head(15), use_container_width=True, hide_index=True)

    st.markdown("&nbsp;")
    if st.button("➡️ Jalankan Preprocessing", type="primary"):
        st.session_state.clean_result = preprocess(raw_df)
        st.session_state.agg_df = aggregate_by_customer(st.session_state.clean_result["data"])
        st.success("Preprocessing selesai — lihat tab **2️⃣ Preprocessing**.")

# ---------------- TAB 2: PREPROCESSING ----------------
with tab_prep:
    st.subheader("Preprocessing & Agregasi Data")
    st.caption("Data mentah dibersihkan (duplikat & nilai kosong ditangani), lalu diagregasi per pelanggan "
               "menjadi tiga variabel: frekuensi pembelian, jumlah transaksi, total nilai pembelian.")
    if "clean_result" not in st.session_state:
        st.info("Jalankan langkah Preprocessing terlebih dahulu di tab **1️⃣ Data**.")
    else:
        cr = st.session_state.clean_result
        agg_df = st.session_state.agg_df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Mentah", cr["before"])
        c2.metric("Data Bersih", cr["after"])
        c3.metric("Baris Dihapus", cr["before"] - cr["after"])
        c4.metric("Pelanggan Unik", len(agg_df))
        st.markdown("**Log Preprocessing**")
        for line in cr["log"]:
            st.markdown(f"✅ {line}")
        st.markdown("**Hasil Agregasi per Pelanggan (pratinjau)**")
        preview = agg_df.copy()
        preview["total_nilai_pembelian"] = preview["total_nilai_pembelian"].apply(fmt_idr)
        st.dataframe(preview.head(10), use_container_width=True, hide_index=True)

        if st.button("➡️ Jalankan Normalisasi", type="primary"):
            norm_df, mins, maxs = normalize_minmax(agg_df)
            st.session_state.norm_df = norm_df
            st.session_state.mins = mins
            st.session_state.maxs = maxs
            st.success("Normalisasi selesai — lihat tab **3️⃣ Normalisasi**.")

# ---------------- TAB 3: NORMALISASI ----------------
with tab_norm:
    st.subheader("Normalisasi Min-Max")
    st.caption("Setiap variabel memiliki rentang nilai berbeda sehingga dinormalisasi ke rentang 0–1 agar "
               "berkontribusi seimbang dalam perhitungan jarak Euclidean pada K-Means.")
    if "norm_df" not in st.session_state:
        st.info("Jalankan langkah Normalisasi terlebih dahulu di tab **2️⃣ Preprocessing**.")
    else:
        st.code("x' = (x - min) / (max - min)", language="text")
        mins, maxs = st.session_state.mins, st.session_state.maxs
        cols = st.columns(3)
        for i, fk in enumerate(FEATURE_KEYS):
            with cols[i]:
                st.markdown(f"**{FEATURE_LABELS[fk]}**")
                if fk == "total_nilai_pembelian":
                    st.write(f"Min: {fmt_idr(mins[fk])}")
                    st.write(f"Max: {fmt_idr(maxs[fk])}")
                else:
                    st.write(f"Min: {mins[fk]}")
                    st.write(f"Max: {maxs[fk]}")

        st.markdown("**Data Setelah Normalisasi (pratinjau)**")
        norm_df = st.session_state.norm_df
        show_cols = ["nama_pelanggan"] + [f"n_{fk}" for fk in FEATURE_KEYS]
        preview = norm_df[show_cols].head(10).round(3)
        preview.columns = ["Nama"] + [f"n({FEATURE_LABELS[fk]})" for fk in FEATURE_KEYS]
        st.dataframe(preview, use_container_width=True, hide_index=True)

        st.markdown("&nbsp;")
        st.info("Lanjutkan ke tab **4️⃣ Elbow & Clustering** untuk menentukan jumlah cluster optimal.")

# ---------------- TAB 4: ELBOW + SILHOUETTE + CLUSTERING ----------------
with tab_cluster:
    st.subheader("Evaluasi (Elbow Method + Silhouette Score) & Clustering K-Means")
    st.caption("Grafik menampilkan WCSS (Elbow Method) dan Silhouette Score berdampingan pada satu grafik "
               "(dua sumbu-Y), lengkap dengan garis bantu vertikal untuk titik siku optimal dan k dengan "
               "Silhouette Score terbaik.")
    if "norm_df" not in st.session_state:
        st.info("Selesaikan tahap Normalisasi terlebih dahulu di tab **3️⃣ Normalisasi**.")
    else:
        norm_df = st.session_state.norm_df
        points = norm_df[[f"n_{fk}" for fk in FEATURE_KEYS]].to_numpy()
        max_k_possible = min(10, len(points) - 1)

        colA, colB = st.columns([1, 3])
        with colA:
            max_k = st.slider("Rentang maksimum k untuk evaluasi", 3, max_k_possible, min(8, max_k_possible))
            run_eval = st.button("📈 Jalankan Elbow & Silhouette", type="primary", use_container_width=True)

        if run_eval:
            with st.spinner("Menghitung WCSS dan Silhouette Score untuk tiap k..."):
                wcss_series, sil_series = compute_wcss_silhouette(points, max_k)
                elbow_k = detect_elbow(wcss_series)
                sil_k = best_silhouette_k(sil_series)
            st.session_state.wcss_series = wcss_series
            st.session_state.sil_series = sil_series
            st.session_state.elbow_k = elbow_k
            st.session_state.sil_k = sil_k

        if "wcss_series" in st.session_state:
            wcss_series = st.session_state.wcss_series
            sil_series = st.session_state.sil_series
            elbow_k = st.session_state.elbow_k
            sil_k = st.session_state.sil_k

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            ks_wcss = [p["k"] for p in wcss_series]
            wcss_vals = [p["wcss"] for p in wcss_series]
            ks_sil = [p["k"] for p in sil_series]
            sil_vals = [p["silhouette"] for p in sil_series]

            fig.add_trace(go.Scatter(x=ks_wcss, y=wcss_vals, name="WCSS (Elbow)",
                                      mode="lines+markers", line=dict(color=ROSE_DEEP, width=3),
                                      marker=dict(size=8)), secondary_y=False)
            fig.add_trace(go.Scatter(x=ks_sil, y=sil_vals, name="Silhouette Score",
                                      mode="lines+markers", line=dict(color=SAGE_DEEP, width=3, dash="dot"),
                                      marker=dict(size=8, symbol="diamond")), secondary_y=True)

            if elbow_k is not None:
                fig.add_vline(x=elbow_k, line_width=2, line_dash="dash", line_color=GOLD,
                               annotation_text=f"Siku optimal k={elbow_k}", annotation_position="top left",
                               annotation_font_color=GOLD)
            if sil_k is not None:
                fig.add_vline(x=sil_k, line_width=2, line_dash="dash", line_color="#2F6F8F",
                               annotation_text=f"Silhouette terbaik k={sil_k}", annotation_position="top right",
                               annotation_font_color="#2F6F8F")

            fig.update_layout(
                height=440, plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, l=10, r=10, b=10),
            )
            fig.update_xaxes(title_text="Jumlah Cluster (k)", dtick=1, gridcolor=LINE)
            fig.update_yaxes(title_text="WCSS", secondary_y=False, gridcolor=LINE)
            fig.update_yaxes(title_text="Silhouette Score", secondary_y=True, gridcolor=LINE)
            st.plotly_chart(fig, use_container_width=True)

            note = []
            if elbow_k is not None:
                note.append(f"**Elbow Method** merekomendasikan **k = {elbow_k}** (penurunan WCSS melandai setelah titik ini).")
            if sil_k is not None:
                best_sil_val = max(s["silhouette"] for s in sil_series if not np.isnan(s["silhouette"]))
                note.append(f"**Silhouette Score** tertinggi ada pada **k = {sil_k}** (skor = {best_sil_val:.3f}) — semakin mendekati 1 semakin baik pemisahan antar cluster.")
            st.markdown("  \n".join(note))

            st.markdown("---")
            st.markdown("**Tentukan jumlah cluster (k) final untuk K-Means**")
            default_k = sil_k or elbow_k or 3
            k_final = st.slider("Jumlah cluster (k)", 2, max_k, int(default_k), key="k_final_slider")
            st.caption(f"Rekomendasi Elbow: k={elbow_k} · Rekomendasi Silhouette: k={sil_k}")

            if st.button("🚀 Jalankan K-Means & Lihat Hasil", type="primary"):
                with st.spinner("Menjalankan K-Means..."):
                    labels, centroids_norm, inertia = run_kmeans(points, k_final)
                    centroids_raw = []
                    for c in centroids_norm:
                        obj = {}
                        for i, fk in enumerate(FEATURE_KEYS):
                            rng_ = st.session_state.maxs[fk] - st.session_state.mins[fk]
                            obj[fk] = st.session_state.mins[fk] + c[i] * rng_
                        centroids_raw.append(obj)
                    sizes = [int((labels == c).sum()) for c in range(k_final)]
                    interp = interpret_clusters(centroids_raw, sizes)
                    result_df = st.session_state.agg_df.copy().reset_index(drop=True)
                    result_df["cluster"] = labels
                    result_df["segmen"] = result_df["cluster"].map(lambda c: interp[c]["label"])
                    sil_final = silhouette_score(points, labels) if k_final >= 2 else None

                    st.session_state.cluster_result = {
                        "k": k_final, "labels": labels, "centroids_raw": centroids_raw,
                        "sizes": sizes, "interp": interp, "df": result_df,
                        "inertia": inertia, "silhouette": sil_final,
                    }
                st.success("Clustering selesai — lihat tab **5️⃣ Hasil & Interpretasi AI**.")
        else:
            st.info("Klik **Jalankan Elbow & Silhouette** untuk melihat grafik evaluasi.")

# ---------------- TAB 5: HASIL & INTERPRETASI AI ----------------
with tab_result:
    st.subheader("Hasil Segmentasi Pelanggan")
    if "cluster_result" not in st.session_state:
        st.info("Selesaikan tahap Clustering terlebih dahulu di tab **4️⃣ Elbow & Clustering**.")
    else:
        cres = st.session_state.cluster_result
        k = cres["k"]
        df = cres["df"]
        interp = cres["interp"]
        centroids_raw = cres["centroids_raw"]
        sizes = cres["sizes"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Jumlah Cluster (k)", k)
        m2.metric("Total Pelanggan", len(df))
        m3.metric("Silhouette Score", f"{cres['silhouette']:.3f}" if cres["silhouette"] else "-")

        st.markdown("### Profil Segmen")
        order = sorted(interp.items(), key=lambda kv: kv[1]["rank"])
        cols = st.columns(len(order))
        for col, (idx, info) in zip(cols, order):
            c = centroids_raw[idx]
            color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
            with col:
                st.markdown(
                    f"""
                    <div class="card" style="border-top:4px solid {color};">
                    <strong>{info['label']}</strong><br/>
                    <span style="font-size:12px;color:{INK_SOFT};">{sizes[idx]} pelanggan</span>
                    <p style="font-size:13px;color:{INK_SOFT};">{info['desc']}</p>
                    <div style="font-size:12.5px;">
                    Rata-rata frekuensi: <strong>{c['frekuensi_pembelian']:.1f}x</strong><br/>
                    Rata-rata jumlah transaksi: <strong>{c['jumlah_transaksi']:.1f}</strong><br/>
                    Rata-rata nilai pembelian: <strong>{fmt_idr(c['total_nilai_pembelian'])}</strong>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Visualisasi Cluster")
        fig2 = go.Figure()
        for idx in range(k):
            sub = df[df["cluster"] == idx]
            fig2.add_trace(go.Scatter(
                x=sub["frekuensi_pembelian"], y=sub["total_nilai_pembelian"],
                mode="markers", name=interp[idx]["label"],
                marker=dict(size=9, color=CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)], opacity=0.85),
                text=sub["nama_pelanggan"],
                hovertemplate="<b>%{text}</b><br>Frekuensi: %{x}<br>Nilai: Rp %{y:,.0f}<extra></extra>",
            ))
        fig2.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_title="Frekuensi Pembelian", yaxis_title="Total Nilai Pembelian",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig2.update_xaxes(gridcolor=LINE)
        fig2.update_yaxes(gridcolor=LINE)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Tabel Hasil Segmentasi")
        show_df = df[["nama_pelanggan", "frekuensi_pembelian", "jumlah_transaksi", "total_nilai_pembelian", "segmen"]].copy()
        show_df = show_df.sort_values(by=["segmen"], key=lambda s: s.map(lambda l: [v["rank"] for v in interp.values() if v["label"] == l][0]))
        show_df["total_nilai_pembelian"] = show_df["total_nilai_pembelian"].apply(fmt_idr)
        show_df.columns = ["Nama Pelanggan", "Frekuensi", "Jumlah Transaksi", "Total Nilai", "Segmen"]
        st.dataframe(show_df, use_container_width=True, hide_index=True, height=340)

        csv_export = df[["nama_pelanggan", "no_telepon", "frekuensi_pembelian", "jumlah_transaksi",
                          "total_nilai_pembelian", "cluster", "segmen"]]
        st.download_button("⬇️ Unduh Hasil (CSV)", csv_export.to_csv(index=False).encode("utf-8"),
                            file_name="hasil_segmentasi_pelanggan.csv", mime="text/csv")

        # ============================================================
        # INTERPRETASI & REKOMENDASI STRATEGI OLEH AI (GOOGLE GEMINI)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🤖 Interpretasi & Rekomendasi Strategi oleh AI (Google Gemini)")
        st.caption("Masukkan API Key Google AI Studio Anda untuk memuat daftar model yang tersedia, lalu pilih "
                   "model untuk menganalisis hasil segmentasi dan menghasilkan rekomendasi strategi pemasaran. "
                   "API Key hanya digunakan selama sesi ini dan tidak disimpan.")

        api_key = st.text_input("Google AI API Key", type="password",
                                 help="Dapatkan API Key gratis di https://aistudio.google.com/apikey")

        col_load, col_model = st.columns([1, 2])
        with col_load:
            load_models_clicked = st.button("🔎 Muat Daftar Model", disabled=not api_key)

        if load_models_clicked:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                models = []
                for m in client.models.list():
                    actions = getattr(m, "supported_actions", None) or []
                    if "generateContent" in actions:
                        models.append(m.name.replace("models/", ""))
                if not models:
                    st.warning("Tidak ditemukan model yang mendukung generateContent untuk API Key ini.")
                else:
                    st.session_state.gemini_models = sorted(set(models))
                    st.success(f"{len(st.session_state.gemini_models)} model ditemukan.")
            except Exception as e:
                st.error(f"Gagal memuat daftar model: {e}")

        selected_model = None
        with col_model:
            if "gemini_models" in st.session_state:
                preferred = [m for m in st.session_state.gemini_models if "flash" in m or "pro" in m]
                options = preferred if preferred else st.session_state.gemini_models
                selected_model = st.selectbox("Pilih Model", options)
            else:
                st.selectbox("Pilih Model", ["(muat daftar model terlebih dahulu)"], disabled=True)

        run_ai = st.button("✨ Analisis dengan AI & Rekomendasikan Strategi",
                            type="primary", disabled=not (api_key and selected_model))

        if run_ai:
            summary_lines = []
            for idx, info in order:
                c = centroids_raw[idx]
                summary_lines.append(
                    f"- {info['label']} ({sizes[idx]} pelanggan): rata-rata frekuensi pembelian "
                    f"{c['frekuensi_pembelian']:.1f}x, rata-rata jumlah transaksi {c['jumlah_transaksi']:.1f}, "
                    f"rata-rata total nilai pembelian {fmt_idr(c['total_nilai_pembelian'])}."
                )
            prompt = f"""
Anda adalah konsultan strategi pemasaran untuk sebuah toko buket bunga bernama "Queen Buket Majalaya".
Berikut hasil segmentasi pelanggan menggunakan algoritma K-Means (k={k}) berdasarkan tiga variabel:
frekuensi pembelian, jumlah transaksi, dan total nilai pembelian.

Ringkasan tiap segmen:
{chr(10).join(summary_lines)}

Silhouette Score hasil clustering: {cres['silhouette']:.3f} (skala -1 s.d. 1, semakin tinggi semakin baik pemisahan antar cluster).

Tolong berikan dalam Bahasa Indonesia:
1. Interpretasi singkat karakteristik masing-masing segmen pelanggan di atas.
2. Rekomendasi strategi pemasaran yang konkret dan dapat dieksekusi untuk masing-masing segmen (contoh: jenis promosi, channel komunikasi, program loyalitas, frekuensi kontak).
3. Prioritas segmen mana yang paling penting untuk ditindaklanjuti terlebih dahulu, beserta alasannya.
4. Satu catatan singkat mengenai kualitas hasil clustering berdasarkan nilai Silhouette Score di atas.

Format jawaban dengan judul per segmen menggunakan heading markdown.
"""
            try:
                from google import genai
                with st.spinner(f"Model {selected_model} sedang menganalisis..."):
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(model=selected_model, contents=prompt)
                st.session_state.ai_result = response.text
            except Exception as e:
                st.error(f"Gagal menghasilkan interpretasi AI: {e}")

        if "ai_result" in st.session_state:
            st.markdown(st.session_state.ai_result)
            st.download_button("⬇️ Unduh Rekomendasi (TXT)", st.session_state.ai_result.encode("utf-8"),
                                file_name="rekomendasi_strategi_ai.txt", mime="text/plain")

        st.markdown("---")
        if st.button("♻️ Mulai Ulang dengan Data Baru"):
            st.session_state.raw_df = generate_synthetic_transactions(seed=random.randint(0, 999999))
            st.session_state.data_source = "sintetis"
            reset_pipeline_state()
            st.rerun()
