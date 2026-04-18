"""
Streamlit Dashboard
====================
Interactive visualisation for forecast vs actual demand,
anomaly detection, model metrics, and system monitoring.

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src importable from dashboard/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
METRICS_PATH = Path("artifacts/metrics/evaluation_results.json")
ANOMALY_CSV = Path("artifacts/metrics/test_anomalies.csv")
DATA_PATH = Path("data/raw/synthetic_demand.csv")

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d5986);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-title { font-size: 14px; opacity: 0.8; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .anomaly-high { background-color: #fee2e2; border-left: 4px solid #ef4444; }
    .anomaly-medium { background-color: #fef3c7; border-left: 4px solid #f59e0b; }
    .anomaly-low { background-color: #d1fae5; border-left: 4px solid #10b981; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_local_data() -> Optional[pd.DataFrame]:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        return df
    return None


@st.cache_data(ttl=60)
def load_evaluation_results() -> Optional[dict]:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=60)
def load_anomaly_data() -> Optional[pd.DataFrame]:
    if ANOMALY_CSV.exists():
        df = pd.read_csv(ANOMALY_CSV, parse_dates=["timestamp"])
        return df
    return None


def call_api(endpoint: str, method: str = "GET", payload: dict = None) -> Optional[dict]:
    """Call the API with error handling."""
    try:
        url = f"{API_URL}/{endpoint.lstrip('/')}"
        if method == "GET":
            resp = requests.get(url, timeout=5)
        else:
            resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as exc:
        st.warning(f"API call failed: {exc}")
        return None


def metric_card(title: str, value: str, delta: Optional[str] = None) -> str:
    delta_html = f"<div style='font-size:12px;color:#88cc88'>{delta}</div>" if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("⚡ Demand Forecast")
    st.caption("Intelligent ML Monitoring Dashboard")

    st.divider()
    st.subheader("🔧 Controls")

    # Date range filter
    df_raw = load_local_data()
    if df_raw is not None:
        min_date = df_raw["timestamp"].min().date()
        max_date = df_raw["timestamp"].max().date()
    else:
        min_date = datetime(2021, 1, 1).date()
        max_date = datetime(2023, 1, 1).date()

    default_start = max_date - timedelta(days=30)
    date_range = st.date_input(
        "Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    lookback_days = st.slider("Lookback (days)", 7, 365, 30)
    show_anomalies = st.checkbox("Show Anomalies", value=True)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

    st.divider()
    # API Status
    health = call_api("/health")
    if health:
        status_icon = "🟢" if health.get("status") == "ok" else "🟡"
        st.success(f"{status_icon} API Online")
        st.caption(f"Model: {'✅ Loaded' if health.get('model_loaded') else '❌ Not loaded'}")
        st.caption(f"DB: {'✅' if health.get('db_connected') else '❌'}")
        st.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
    else:
        st.error("🔴 API Offline")
        st.caption("Start with: uvicorn src.api.main:app")

    if auto_refresh:
        time.sleep(60)
        st.rerun()


# ── Main Tabs ─────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast", "🚨 Anomalies", "📊 Model Metrics", "🔍 Live Predict"
])


# ── Tab 1: Forecast ───────────────────────────────────────────────────

with tab1:
    st.header("Demand Forecast vs Actual")

    df = load_local_data()
    anomaly_df = load_anomaly_data()

    if df is None:
        st.warning("No data found. Run the pipeline first: `python -m src.pipeline`")
    else:
        # Filter by date range
        if len(date_range) == 2:
            start_dt, end_dt = date_range
            mask = (df["timestamp"].dt.date >= start_dt) & (df["timestamp"].dt.date <= end_dt)
            df_view = df[mask].copy()
        else:
            df_view = df.tail(lookback_days * 24).copy()

        # Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                metric_card("Avg Demand", f"{df_view['demand'].mean():.0f} kWh"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                metric_card("Peak Demand", f"{df_view['demand'].max():.0f} kWh"),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                metric_card("Min Demand", f"{df_view['demand'].min():.0f} kWh"),
                unsafe_allow_html=True,
            )
        with col4:
            n_anom = len(anomaly_df[anomaly_df["is_anomaly"]]) if anomaly_df is not None else 0
            st.markdown(
                metric_card("Anomalies", str(n_anom), "⚠️ from test set"),
                unsafe_allow_html=True,
            )

        st.subheader("Time Series")

        # Resample for performance
        if len(df_view) > 2000:
            df_plot = df_view.resample("6H", on="timestamp")["demand"].mean().reset_index()
        else:
            df_plot = df_view

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["timestamp"],
            y=df_plot["demand"],
            mode="lines",
            name="Actual Demand",
            line={"color": "#2196F3", "width": 1.5},
        ))

        # Overlay anomalies
        if show_anomalies and anomaly_df is not None:
            anom_pts = anomaly_df[
                anomaly_df["is_anomaly"] &
                anomaly_df["timestamp"].notna()
            ]
            if len(anom_pts) > 0:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(anom_pts["timestamp"]),
                    y=anom_pts["actual"],
                    mode="markers",
                    name="Anomaly",
                    marker={"color": "#EF4444", "size": 8, "symbol": "x"},
                ))

        fig.update_layout(
            height=450,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis={"showgrid": True, "gridcolor": "#eee"},
            yaxis={"showgrid": True, "gridcolor": "#eee", "title": "Demand (kWh)"},
            legend={"orientation": "h", "y": 1.02},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.subheader("Hourly Demand Heatmap")
        df_hm = df_view.copy()
        df_hm["hour"] = df_hm["timestamp"].dt.hour
        df_hm["day"] = df_hm["timestamp"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = df_hm.pivot_table(values="demand", index="hour", columns="day", aggfunc="mean")
        pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])

        fig_hm = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels={"x": "Day", "y": "Hour", "color": "kWh"},
        )
        fig_hm.update_layout(height=350)
        st.plotly_chart(fig_hm, use_container_width=True)


# ── Tab 2: Anomalies ──────────────────────────────────────────────────

with tab2:
    st.header("🚨 Anomaly Detection")
    anomaly_df = load_anomaly_data()

    if anomaly_df is None:
        st.info("No anomaly data found. Run the training pipeline first.")
    else:
        detected = anomaly_df[anomaly_df["is_anomaly"]]

        c1, c2, c3, c4 = st.columns(4)
        severity_counts = detected["severity"].value_counts()

        with c1:
            st.metric("Total Anomalies", len(detected))
        with c2:
            st.metric("High Severity", severity_counts.get("high", 0), delta="⚠️")
        with c3:
            st.metric("Medium Severity", severity_counts.get("medium", 0))
        with c4:
            st.metric("Low Severity", severity_counts.get("low", 0))

        st.subheader("Anomaly Score Distribution")
        fig_score = px.histogram(
            anomaly_df,
            x="anomaly_score",
            color="is_anomaly",
            nbins=50,
            color_discrete_map={True: "#EF4444", False: "#22C55E"},
            labels={"anomaly_score": "Anomaly Score", "is_anomaly": "Anomaly"},
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig_score, use_container_width=True)

        st.subheader("Anomaly Timeline")
        if "timestamp" in detected.columns and detected["timestamp"].notna().any():
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=anomaly_df["timestamp"],
                y=anomaly_df["actual"],
                mode="lines",
                name="Demand",
                line={"color": "#94a3b8", "width": 1},
                opacity=0.6,
            ))

            color_map = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}
            for sev, color in color_map.items():
                pts = detected[detected["severity"] == sev]
                if len(pts) > 0:
                    fig_t.add_trace(go.Scatter(
                        x=pts["timestamp"],
                        y=pts["actual"],
                        mode="markers",
                        name=f"{sev.title()} anomaly",
                        marker={"color": color, "size": 9, "symbol": "circle-open"},
                    ))

            fig_t.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_t, use_container_width=True)

        st.subheader("Anomaly Records")
        st.dataframe(
            detected.sort_values("anomaly_score", ascending=False)
            .head(50)
            .reset_index(drop=True),
            use_container_width=True,
        )


# ── Tab 3: Model Metrics ──────────────────────────────────────────────

with tab3:
    st.header("📊 Model Performance")
    eval_results = load_evaluation_results()

    # Try API first
    api_metrics = call_api("/metrics")

    if api_metrics:
        st.success("Metrics from live API")
        cols = st.columns(4)
        metrics_display = [
            ("RMSE", f"{api_metrics.get('test_rmse', 0):.2f} kWh"),
            ("MAE", f"{api_metrics.get('test_mae', 0):.2f} kWh"),
            ("MAPE", f"{api_metrics.get('test_mape', 0):.2f}%"),
            ("R²", f"{api_metrics.get('test_r2', 0):.4f}"),
        ]
        for col, (name, val) in zip(cols, metrics_display):
            with col:
                st.markdown(metric_card(name, val), unsafe_allow_html=True)

        st.info(f"Model: **{api_metrics.get('model_name', 'N/A')}** v{api_metrics.get('model_version', '?')} | "
                f"Trained: {api_metrics.get('last_trained_at', 'N/A')}")

    if eval_results:
        st.subheader("All Models Comparison")
        rows = []
        for name, data in eval_results.items():
            m = data.get("metrics", {})
            rows.append({
                "Model": name,
                "RMSE": round(m.get("test_rmse", 0), 2),
                "MAE": round(m.get("test_mae", 0), 2),
                "MAPE (%)": round(m.get("test_mape", 0), 2),
                "R²": round(m.get("test_r2", 0), 4),
            })
        df_comp = pd.DataFrame(rows).sort_values("RMSE")

        fig_bar = px.bar(
            df_comp,
            x="Model",
            y="RMSE",
            color="RMSE",
            color_continuous_scale="RdYlGn_r",
            title="Test RMSE by Model (lower = better)",
        )
        fig_bar.update_layout(height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(df_comp.set_index("Model"), use_container_width=True)
    else:
        st.info("Run the training pipeline to generate model metrics.")


# ── Tab 4: Live Predict ───────────────────────────────────────────────

with tab4:
    st.header("🔍 Live Demand Prediction")
    st.write("Send a real-time forecast request to the API.")

    col_a, col_b = st.columns(2)
    with col_a:
        target_ts = st.datetime_input("Target Timestamp", value=datetime.now())
        lag_1 = st.number_input("Lag 1h (previous hour demand)", value=850.0, step=10.0)
        lag_24 = st.number_input("Lag 24h (same hour yesterday)", value=820.0, step=10.0)
        lag_168 = st.number_input("Lag 168h (same hour last week)", value=830.0, step=10.0)
    with col_b:
        roll_mean = st.number_input("Rolling Mean 24h", value=840.0, step=10.0)
        roll_std = st.number_input("Rolling Std 24h", value=45.0, step=5.0)
        hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
        dow = st.slider("Day of Week (0=Mon)", 0, 6, datetime.now().weekday())

    if st.button("🚀 Get Forecast", type="primary"):
        if health is None:
            st.error("API is offline. Start the API server first.")
        else:
            import math
            payload = {
                "timestamp": target_ts.isoformat(),
                "features": {
                    "demand_lag_1h": lag_1,
                    "demand_lag_24h": lag_24,
                    "demand_lag_168h": lag_168,
                    "demand_roll_mean_24h": roll_mean,
                    "demand_roll_std_24h": roll_std,
                    "hour": hour,
                    "day_of_week": dow,
                    "month": target_ts.month,
                    "is_weekend": 1 if dow >= 5 else 0,
                    "is_peak_hour": 1 if hour in range(7, 10) or hour in range(17, 22) else 0,
                    "hour_sin": math.sin(2 * math.pi * hour / 24),
                    "hour_cos": math.cos(2 * math.pi * hour / 24),
                    "day_of_week_sin": math.sin(2 * math.pi * dow / 7),
                    "day_of_week_cos": math.cos(2 * math.pi * dow / 7),
                },
            }

            with st.spinner("Querying model..."):
                result = call_api("/predict", method="POST", payload=payload)

            if result:
                st.success("✅ Prediction successful!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Demand", f"{result['predicted_demand']:.1f} kWh")
                col2.metric("Model", result["model_name"])
                col3.metric("Latency", f"{result['latency_ms']:.1f} ms")
                st.json(result)
            else:
                st.error("Prediction failed. Check API logs.")

    st.divider()
    st.subheader("Anomaly Detection")
    actual_val = st.number_input("Actual Demand (kWh)", value=900.0, step=10.0)
    pred_val = st.number_input("Predicted Demand (kWh)", value=850.0, step=10.0)

    if st.button("🔍 Check for Anomaly", type="secondary"):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "actual_demand": actual_val,
            "predicted_demand": pred_val,
            "method": "ensemble",
        }
        with st.spinner("Checking..."):
            result = call_api("/detect-anomaly", method="POST", payload=payload)

        if result:
            if result["is_anomaly"]:
                sev = result["severity"]
                colors = {"high": "error", "medium": "warning", "low": "info"}
                getattr(st, colors.get(sev, "info"))(
                    f"⚠️ **Anomaly Detected!** Severity: {sev.upper()} | "
                    f"Score: {result['anomaly_score']:.3f}"
                )
            else:
                st.success(f"✅ Normal observation (score={result['anomaly_score']:.3f})")
            st.json(result)


# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚡ Intelligent Demand Forecasting & Anomaly Detection System | "
    f"Built with FastAPI + Streamlit | Last updated: {datetime.now().strftime('%H:%M:%S')}"
)
