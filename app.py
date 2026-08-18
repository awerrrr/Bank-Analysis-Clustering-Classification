from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_FILE = DATA_DIR / "data_clustering.csv"

NUM_COLS = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]
CAT_COLS = ["TransactionType", "Location", "Channel", "CustomerOccupation"]

# ========== TOKEN DESAIN ==========
# Tema "ledger" (buku besar bank): navy gelap + satu aksen teal, angka pakai
# font mono biar berasa kayak layar mesin ATM/terminal transaksi -- bukan
# dashboard generik warna-warni.
INK = "#0B1220"
SURFACE = "#141F35"
SURFACE_BORDER = "#26334D"
TEXT = "#E8EDF4"
TEXT_MUTED = "#8B96AC"
ACCENT = "#4FD1C5"
ACCENT_WARM = "#F2A65A"
CATEGORY_COLORS = ["#4FD1C5", "#F2A65A", "#8C7CF0", "#F2789F", "#6FCF97", "#56A8E0", "#E0C56F", "#E07856"]


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background-color: {INK}; color: {TEXT}; }}

        h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}

        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
            color: {ACCENT} !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
        div[data-testid="stMetric"] {{
            background-color: {SURFACE};
            border: 1px solid {SURFACE_BORDER};
            border-top: 2px solid {ACCENT};
            border-radius: 6px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {SURFACE};
            border-color: {SURFACE_BORDER} !important;
            border-radius: 10px;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {SURFACE_BORDER}; }}
        .stTabs [data-baseweb="tab"] {{ color: {TEXT_MUTED}; font-family: 'Space Grotesk', sans-serif; }}
        .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

        .ledger-caption {{ color: {TEXT_MUTED}; font-size: 0.85rem; margin-top: -0.4rem; }}
        section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {SURFACE_BORDER}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def register_plotly_theme() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        title_font=dict(family="Space Grotesk, sans-serif", color=TEXT, size=16),
        colorway=CATEGORY_COLORS,
        xaxis=dict(gridcolor=SURFACE_BORDER, zerolinecolor=SURFACE_BORDER, linecolor=SURFACE_BORDER),
        yaxis=dict(gridcolor=SURFACE_BORDER, zerolinecolor=SURFACE_BORDER, linecolor=SURFACE_BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=48, l=10, r=10, b=10),
    )
    pio.templates["ledger"] = template
    px.defaults.template = "ledger"


# ========== DATA & MODEL (cached) ==========
def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner="Memuat data...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def filter_data(df: pd.DataFrame, locations: list, channels: list, targets: list) -> pd.DataFrame:
    return df[
        df["Location"].isin(locations)
        & df["Channel"].isin(channels)
        & df["Target"].isin(targets)
    ].copy()


@st.cache_data(show_spinner="Menjalankan clustering & PCA...")
def run_clustering(df: pd.DataFrame, k: int):
    cluster_df = df[NUM_COLS].fillna(df[NUM_COLS].median())
    scaled = StandardScaler().fit_transform(cluster_df)

    cluster_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    sil = silhouette_score(scaled, cluster_labels) if len(set(cluster_labels)) > 1 and len(df) > 2 else None

    # PCA asli dari SEMUA fitur numerik (bukan cuma 2 kolom mentah) -- proyeksi
    # 2D ini jadi representasi terbaik dari "jarak" antar transaksi di 5 dimensi.
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": cluster_labels.astype(str)})

    profile = cluster_df.copy()
    profile["cluster"] = cluster_labels.astype(str)
    cluster_profile = profile.groupby("cluster")[NUM_COLS].mean().round(1)
    cluster_profile["jumlah_transaksi"] = profile["cluster"].value_counts().sort_index()

    return plot_df, sil, cluster_profile


@st.cache_data(show_spinner="Menguji beberapa nilai k...")
def silhouette_curve(df: pd.DataFrame, k_min: int, k_max: int) -> pd.DataFrame:
    cluster_df = df[NUM_COLS].fillna(df[NUM_COLS].median())
    scaled = StandardScaler().fit_transform(cluster_df)
    scores = []
    for k in range(k_min, k_max + 1):
        cluster_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        if len(set(cluster_labels)) > 1:
            scores.append((k, silhouette_score(scaled, cluster_labels)))
    return pd.DataFrame(scores, columns=["k", "silhouette"])


@st.cache_resource(show_spinner="Melatih model klasifikasi...")
def train_classifier(df: pd.DataFrame):
    X = df[NUM_COLS + CAT_COLS]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", one_hot(), CAT_COLS),
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLS),
                    ]
                ),
            ),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    classes = pipeline.named_steps["model"].classes_
    report = pd.DataFrame(classification_report(y_test, preds, output_dict=True)).T.round(3)
    cm = confusion_matrix(y_test, preds, labels=classes)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="weighted"),
    }
    return pipeline, X_test, y_test, preds, metrics, report, cm, classes


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    ohe_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_COLS)
    feature_names = list(ohe_names) + NUM_COLS
    importances = pipeline.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )


# ========== PAGE ==========
st.set_page_config(page_title="Bank Analysis", page_icon="🏦", layout="wide")
inject_theme()
register_plotly_theme()

st.title("🏦 Bank Transaction Analysis")
st.caption("Segmentasi nasabah (K-Means) + prediksi target (Random Forest) dari data transaksi bank.")
with st.expander("Tentang project ini"):
    st.markdown(
        """
        Alurnya dua tahap: **Clustering** dulu buat ngelompokin transaksi jadi beberapa
        segmen berdasarkan pola nominal, usia, durasi, login attempt, dan saldo — tanpa label.
        Hasil segmen itu lalu jadi target (`Target`) yang diprediksi ulang pakai **Random Forest**
        di tab Model, biar pola yang ditemukan clustering bisa dijelaskan/diprediksi otomatis
        untuk transaksi baru (lihat tab **What-if**).
        """
    )

if not DATA_FILE.exists():
    st.warning("Dataset belum ada. Taruh `data_clustering.csv` di dalam folder `data/`.")
    st.stop()

df = load_data()

with st.sidebar.form("filter_form"):
    st.header("Filter")
    location_filter = st.multiselect(
        "Location", sorted(df["Location"].dropna().unique()),
        default=sorted(df["Location"].dropna().unique())[:8],
    )
    channel_filter = st.multiselect(
        "Channel", sorted(df["Channel"].dropna().unique()),
        default=sorted(df["Channel"].dropna().unique()),
    )
    target_filter = st.multiselect(
        "Target", sorted(df["Target"].dropna().unique()),
        default=sorted(df["Target"].dropna().unique()),
    )
    apply_filters = st.form_submit_button("Terapkan Filter", use_container_width=True)

# Filter dibungkus st.form supaya clustering & training model tidak jalan ulang
# tiap satu checkbox digeser -- baru diproses saat tombol "Terapkan Filter" ditekan.
if "bank_filters" not in st.session_state or apply_filters:
    st.session_state.bank_filters = (tuple(location_filter), tuple(channel_filter), tuple(target_filter))

loc_f, chan_f, targ_f = st.session_state.bank_filters
filtered = filter_data(df, list(loc_f), list(chan_f), list(targ_f))
st.sidebar.caption(f"📌 {len(filtered):,} dari {len(df):,} baris cocok dengan filter ini.".replace(",", "."))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(filtered):,}".replace(",", "."), help="Jumlah baris transaksi setelah filter diterapkan.")
col2.metric("Avg Amount", f"{filtered['TransactionAmount'].mean():.2f}" if not filtered.empty else "-", help="Rata-rata nominal transaksi.")
col3.metric("Target Classes", f"{filtered['Target'].nunique()}", help="Jumlah kelas target unik pada data terfilter.")
col4.metric("Top Channel", filtered["Channel"].mode().iat[0] if not filtered.empty else "-", help="Channel transaksi paling sering dipakai.")

tab_overview, tab_cluster, tab_model, tab_whatif, tab_data = st.tabs(
    ["📊 Overview", "🧩 Clustering", "🌲 Model", "🔮 What-if", "📄 Data"]
)

EMPTY_MSG = "Belum ada data yang cocok dengan filter ini. Coba longgarkan pilihan Location/Channel/Target di sidebar."

with tab_overview:
    if filtered.empty:
        st.info(EMPTY_MSG)
    else:
        with st.container(border=True):
            left, right = st.columns(2)
            with left:
                fig = px.histogram(filtered, x="TransactionAmount", nbins=40, title="Sebaran Transaction Amount")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                top_loc = filtered["Location"].value_counts().head(10).reset_index()
                fig = px.bar(top_loc, x="count", y="Location", orientation="h", title="Top 10 Location")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            st.markdown("**Korelasi antar fitur numerik**")
            corr = filtered[NUM_COLS].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale=["#141F35", ACCENT], aspect="auto")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<p class="ledger-caption">Semakin dekat ke 1 (terang), semakin kuat dua fitur ini bergerak bersamaan — berguna buat cek fitur mana yang redundan sebelum dipakai clustering/model.</p>',
                unsafe_allow_html=True,
            )

with tab_cluster:
    if filtered.empty:
        st.info(EMPTY_MSG)
    else:
        with st.container(border=True):
            k = st.slider("Jumlah cluster (k)", 2, 8, min(3, max(2, len(filtered) // 150)))
            plot_df, sil, cluster_profile = run_clustering(filtered, k)

            m1, m2 = st.columns(2)
            if sil is not None:
                m1.metric("Silhouette score", f"{sil:.3f}", help="Mendekati 1 = cluster terpisah rapi; mendekati 0 = tumpang tindih.")
            m2.metric("Jumlah cluster", k)

            fig = px.scatter(plot_df, x="x", y="y", color="cluster", title="Peta cluster (proyeksi PCA 2D)")
            fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<p class="ledger-caption">Sumbu x/y bukan fitur asli — ini rangkuman dari 5 fitur numerik sekaligus (PCA), biar bisa digambar di 2D. Titik yang berdekatan = transaksinya mirip.</p>',
                unsafe_allow_html=True,
            )

            if st.button("🔍 Cari k terbaik (uji k=2..8)"):
                curve = silhouette_curve(filtered, 2, 8)
                if not curve.empty:
                    fig = px.line(curve, x="k", y="silhouette", markers=True, title="Silhouette score per k")
                    st.plotly_chart(fig, use_container_width=True)
                    best_k = int(curve.loc[curve["silhouette"].idxmax(), "k"])
                    st.caption(f"k dengan silhouette tertinggi: **{best_k}**")

        with st.container(border=True):
            st.markdown("**Profil tiap cluster** (rata-rata fitur numerik)")
            st.dataframe(cluster_profile, use_container_width=True)
            st.markdown(
                '<p class="ledger-caption">Ini yang bikin cluster "bisa dibaca" secara bisnis — misal cluster dengan AccountBalance tertinggi bisa jadi target program nasabah prioritas.</p>',
                unsafe_allow_html=True,
            )

with tab_model:
    if len(filtered) < 20:
        st.info("Perlu minimal 20 baris data setelah filter buat melatih model.")
    else:
        pipeline, X_test, y_test, preds, metrics, report, cm, classes = train_classifier(filtered)

        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            c2.metric("Weighted F1", f"{metrics['f1']:.3f}")

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Confusion matrix**")
                fig = px.imshow(
                    cm, x=[str(c) for c in classes], y=[str(c) for c in classes],
                    text_auto=True, color_continuous_scale=["#141F35", ACCENT],
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.markdown("**Classification report**")
                st.dataframe(report, use_container_width=True)

        with st.container(border=True):
            st.markdown("**Feature importance**")
            importance_df = get_feature_importance(pipeline)
            fig = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Top 15 fitur paling berpengaruh")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

with tab_whatif:
    if len(filtered) < 20:
        st.info("Perlu minimal 20 baris data setelah filter buat melatih model prediksi.")
    else:
        pipeline, *_ = train_classifier(filtered)
        st.write("Masukkan data transaksi buat lihat prediksi model secara langsung.")
        with st.form("whatif_form"):
            wc1, wc2 = st.columns(2)
            with wc1:
                amount = st.number_input("Transaction Amount", value=float(filtered["TransactionAmount"].median()))
                age = st.number_input("Customer Age", value=float(filtered["CustomerAge"].median()))
                duration = st.number_input("Transaction Duration", value=float(filtered["TransactionDuration"].median()))
            with wc2:
                login_attempts = st.number_input("Login Attempts", value=float(filtered["LoginAttempts"].median()))
                balance = st.number_input("Account Balance", value=float(filtered["AccountBalance"].median()))
            tc1, tc2 = st.columns(2)
            with tc1:
                trans_type = st.selectbox("Transaction Type", sorted(filtered["TransactionType"].dropna().unique()))
                location = st.selectbox("Location", sorted(filtered["Location"].dropna().unique()))
            with tc2:
                channel = st.selectbox("Channel", sorted(filtered["Channel"].dropna().unique()))
                occupation = st.selectbox("Customer Occupation", sorted(filtered["CustomerOccupation"].dropna().unique()))
            predict_clicked = st.form_submit_button("Prediksi", use_container_width=True)

        if predict_clicked:
            input_row = pd.DataFrame([{
                "TransactionAmount": amount,
                "CustomerAge": age,
                "TransactionDuration": duration,
                "LoginAttempts": login_attempts,
                "AccountBalance": balance,
                "TransactionType": trans_type,
                "Location": location,
                "Channel": channel,
                "CustomerOccupation": occupation,
            }])
            pred = pipeline.predict(input_row)[0]
            proba = pipeline.predict_proba(input_row)[0]
            classes = pipeline.named_steps["model"].classes_
            proba_df = pd.DataFrame({"target": classes, "probability": proba}).sort_values("probability", ascending=True)

            with st.container(border=True):
                st.success(f"Prediksi Target: **{pred}**")
                fig = px.bar(
                    proba_df, x="probability", y="target", orientation="h",
                    title="Probabilitas tiap kelas", range_x=[0, 1],
                )
                fig.update_traces(marker_color=[ACCENT if t == pred else SURFACE_BORDER for t in proba_df["target"]])
                st.plotly_chart(fig, use_container_width=True)

with tab_data:
    with st.container(border=True):
        st.dataframe(filtered.head(250), use_container_width=True)
        st.download_button(
            "⬇️ Download data terfilter (CSV)",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="bank_filtered.csv",
            mime="text/csv",
        )
    with st.expander("Ringkasan statistik (describe)"):
        st.dataframe(filtered[NUM_COLS].describe().round(2), use_container_width=True)
