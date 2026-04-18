from __future__ import annotations
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Fix import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ================= CONFIG =================
st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="⚡",
    layout="wide",
)

# 🚫 Disable API for Streamlit Cloud
API_URL = None

DATA_PATH = Path("data/raw/synthetic_demand.csv")
ANOMALY_CSV = Path("artifacts/metrics/test_anomalies.csv")
METRICS_PATH = Path("artifacts/metrics/evaluation_results.json")

# ================= HELPERS =================


@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return None


@st.cache_data
def load_anomalies() -> Optional[pd.DataFrame]:
    if ANOMALY_CSV.exists():
        return pd.read_csv(ANOMALY_CSV, parse_dates=["timestamp"])
    return None


@st.cache_data
def load_metrics() -> Optional[dict]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def call_api(*args, **kwargs):
    return None  # Disabled


# ================= SIDEBAR =================

st.sidebar.title("⚡ Demand Forecast")
st.sidebar.caption("Streamlit Deployment Version")

df = load_data()

if df is not None:
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
else:
    min_date = datetime(2022, 1, 1).date()
    max_date = datetime(2023, 1, 1).date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(max_date - timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date,
)

# ================= MAIN =================

st.title("⚡ Demand Forecast Dashboard")

# ---------- LOAD DATA ----------
df = load_data()
anom = load_anomalies()

if df is None:
    st.error("❌ Data not found. Please check your repo paths.")
    st.stop()

# ---------- FILTER ----------
start, end = date_range
df = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)

col1.metric("Avg Demand", f"{df['demand'].mean():.0f}")
col2.metric("Max Demand", f"{df['demand'].max():.0f}")
col3.metric("Min Demand", f"{df['demand'].min():.0f}")

# ---------- PLOT ----------
st.subheader("Demand Time Series")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["demand"],
    name="Demand",
    mode="lines"
))

# Add anomalies
if anom is not None:
    anom_pts = anom[anom["is_anomaly"] == True]
    fig.add_trace(go.Scatter(
        x=anom_pts["timestamp"],
        y=anom_pts["actual"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=8)
    ))

st.plotly_chart(fig, use_container_width=True)

# ---------- HEATMAP ----------
st.subheader("Hourly Heatmap")

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day_name()

pivot = df.pivot_table(values="demand", index="hour", columns="day")

fig2 = px.imshow(pivot, color_continuous_scale="YlOrRd")
st.plotly_chart(fig2, use_container_width=True)

# ---------- METRICS ----------
st.subheader("Model Metrics")

metrics = load_metrics()

if metrics:
    rows = []
    for name, m in metrics.items():
        mm = m.get("metrics", {})
        rows.append({
            "Model": name,
            "RMSE": mm.get("test_rmse", 0),
            "MAE": mm.get("test_mae", 0),
            "MAPE": mm.get("test_mape", 0),
        })
    st.dataframe(pd.DataFrame(rows))
else:
    st.info("No metrics found")

# ---------- FOOTER ----------
st.caption("⚡ Deployed on Streamlit Cloud (API disabled version)")
