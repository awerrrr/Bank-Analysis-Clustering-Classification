from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_FILE = DATA_DIR / "data_clustering.csv"


def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


st.set_page_config(page_title="Bank Analysis", page_icon="🏦", layout="wide")
st.title("Bank Analysis")
st.caption("Interactive Streamlit version of the bank clustering + classification project.")

if not DATA_FILE.exists():
    st.warning("Dataset not found. Place `data_clustering.csv` inside `data/`.")
    st.stop()

df = pd.read_csv(DATA_FILE)

num_cols = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]
cat_cols = ["TransactionType", "Location", "Channel", "CustomerOccupation"]

st.sidebar.header("Filters")
location_filter = st.sidebar.multiselect("Location", sorted(df["Location"].dropna().unique()), default=sorted(df["Location"].dropna().unique())[:8])
channel_filter = st.sidebar.multiselect("Channel", sorted(df["Channel"].dropna().unique()), default=sorted(df["Channel"].dropna().unique()))
target_filter = st.sidebar.multiselect("Target", sorted(df["Target"].dropna().unique()), default=sorted(df["Target"].dropna().unique()))

filtered = df[
    df["Location"].isin(location_filter)
    & df["Channel"].isin(channel_filter)
    & df["Target"].isin(target_filter)
].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(filtered):,}".replace(",", "."))
col2.metric("Avg Amount", f"{filtered['TransactionAmount'].mean():.2f}")
col3.metric("Target Classes", f"{filtered['Target'].nunique()}")
col4.metric("Top Channel", filtered["Channel"].mode().iat[0] if not filtered.empty else "-")

tab_overview, tab_cluster, tab_model, tab_data = st.tabs(["Overview", "Clustering", "Model", "Data"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x="TransactionAmount", nbins=40, title="Transaction Amount")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(filtered["Location"].value_counts().head(10).reset_index(), x="count", y="Location", orientation="h", title="Top Locations")
        st.plotly_chart(fig, use_container_width=True)

with tab_cluster:
    cluster_df = filtered[num_cols].fillna(filtered[num_cols].median())
    scaled = StandardScaler().fit_transform(cluster_df)
    labels = KMeans(n_clusters=min(3, max(2, len(filtered) // 150)), random_state=42, n_init=10).fit_predict(scaled)
    pca = StandardScaler().fit_transform(cluster_df[["TransactionAmount", "CustomerAge"]])
    plot_df = pd.DataFrame({"x": pca[:, 0], "y": pca[:, 1], "cluster": labels, "target": filtered["Target"].values})
    sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 and len(filtered) > 2 else None
    if sil is not None:
        st.metric("Silhouette", f"{sil:.3f}")
    st.plotly_chart(px.scatter(plot_df, x="x", y="y", color="cluster", title="Cluster preview"), use_container_width=True)

with tab_model:
    X = filtered[num_cols + cat_cols]
    y = filtered["Target"]
    if len(filtered) < 20:
        st.info("Need at least 20 rows after filtering to train a quick benchmark.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipeline = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("cat", one_hot(), cat_cols),
                            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
                        ]
                    ),
                ),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
        c2.metric("Weighted F1", f"{f1_score(y_test, preds, average='weighted'):.3f}")
        st.write("Benchmark classification output")
        st.dataframe(pd.DataFrame({"actual": y_test.reset_index(drop=True), "predicted": preds}), use_container_width=True)

with tab_data:
    st.dataframe(filtered.head(250), use_container_width=True)
