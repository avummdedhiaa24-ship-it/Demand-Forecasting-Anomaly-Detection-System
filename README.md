# ⚡ Intelligent Demand Forecasting & Anomaly Detection System

> A production-grade, end-to-end ML system for electricity demand forecasting and real-time anomaly detection — built with FastAPI, SQLAlchemy, Streamlit, and TensorFlow.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Setup Instructions](#setup-instructions)
5. [Running the Pipeline](#running-the-pipeline)
6. [API Usage](#api-usage)
7. [Dashboard](#dashboard)
8. [Model Explanation](#model-explanation)
9. [Anomaly Detection](#anomaly-detection)
10. [Docker Deployment](#docker-deployment)
11. [Testing](#testing)
12. [Monitoring](#monitoring)
13. [Configuration Reference](#configuration-reference)

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEMAND FORECASTING SYSTEM                         │
│                                                                       │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────────────┐ │
│  │  Raw Data │──▶│ Ingestion &  │──▶│    Feature Engineering       │ │
│  │  (CSV/DB) │   │  Validation  │   │  Lags · Rolling · Calendar   │ │
│  └──────────┘   └──────────────┘   └───────────────┬──────────────┘ │
│                                                     │                 │
│          ┌──────────────────────────────────────────▼──────────────┐ │
│          │                  MODEL REGISTRY                          │ │
│          │  LinearReg │ Ridge │ Lasso │ RF │ XGBoost │ LGBM │ LSTM │ │
│          └──────────────────────────────────────────┬──────────────┘ │
│                                                     │                 │
│    ┌─────────────────┐         ┌────────────────────▼─────────────┐  │
│    │ Anomaly Detectors│         │       Evaluation Engine          │  │
│    │  IsoForest       │         │  RMSE · MAE · MAPE · TS CV       │  │
│    │  Z-Score         │         └────────────────────┬─────────────┘  │
│    │  Residual-based  │                              │                 │
│    │  Ensemble ←──────│──────────────────────────────┘                 │
│    └─────────────────┘                                                │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                      FastAPI Service                          │    │
│  │  POST /predict  │  POST /detect-anomaly  │  GET /metrics      │    │
│  │  POST /predict/batch  │  GET /health  │  GET /*/history       │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                          │                   │                        │
│            ┌─────────────▼──┐   ┌────────────▼─────────┐            │
│            │  PostgreSQL DB  │   │  Streamlit Dashboard  │            │
│            │  (SQLAlchemy)   │   │  Forecast · Anomalies │            │
│            └────────────────┘   └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
demand_forecast/
├── config/
│   └── config.yaml              # All system configuration
├── data/
│   ├── raw/                     # Raw ingested data
│   └── processed/               # Cleaned & feature-engineered data
├── src/
│   ├── ingestion/
│   │   └── data_loader.py       # Download, load, validate raw data
│   ├── preprocessing/
│   │   └── cleaner.py           # Null imputation, outlier treatment, scaling
│   ├── feature_engineering/
│   │   └── features.py          # Lags, rolling stats, calendar, cyclical features
│   ├── models/
│   │   ├── base_model.py        # Abstract base forecaster
│   │   ├── baseline_models.py   # LinearRegression, Ridge, Lasso
│   │   ├── advanced_models.py   # RandomForest, XGBoost, LightGBM
│   │   └── lstm_model.py        # LSTM deep learning model
│   ├── evaluation/
│   │   ├── metrics.py           # RMSE, MAE, MAPE, R², TS cross-validation
│   │   └── eda.py               # Decomposition, ACF/PACF, heatmaps
│   ├── anomaly_detection/
│   │   └── detector.py          # IsolationForest, Z-score, Residual, Ensemble
│   ├── api/
│   │   ├── main.py              # FastAPI application + all endpoints
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── database.py          # SQLAlchemy ORM + repository layer
│   ├── utils/
│   │   ├── config.py            # Config loader with env var substitution
│   │   ├── logger.py            # Structured loguru logger
│   │   ├── helpers.py           # Serialisation, metrics, hashing
│   │   └── monitoring.py        # Latency tracking, drift detection
│   └── pipeline.py              # End-to-end training orchestrator
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   └── test_anomaly.py
│   └── integration/
│       └── test_api.py
├── artifacts/
│   ├── models/                  # Serialised model files
│   ├── plots/                   # Generated EDA + evaluation plots
│   └── metrics/                 # JSON evaluation reports
├── logs/                        # Rotating application logs
├── notebooks/                   # Jupyter exploration notebooks
├── Dockerfile                   # Multi-stage production Docker build
├── docker-compose.yml           # Full stack: API + DB + Dashboard
├── Makefile                     # Convenience commands
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone / enter project
cd demand_forecast

# Install dependencies
make install

# Setup environment
make setup            # creates .env from .env.example

# Run training pipeline (generates synthetic data automatically)
make train

# Start API
make api              # → http://localhost:8000/docs

# Start Dashboard (new terminal)
make dashboard        # → http://localhost:8501
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (or SQLite auto-fallback for local dev)
- Docker + Docker Compose (for containerised deployment)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your database credentials
```

Key variables:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=demand_forecast
DB_USER=postgres
DB_PASSWORD=your_password
API_PORT=8000
DASHBOARD_PORT=8501
```

### 3. Database Setup

The system auto-creates tables on first start. For PostgreSQL:

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE demand_forecast;"
```

---

## 🏋️ Running the Pipeline

### Full Pipeline (recommended)

```bash
make train
# or
PYTHONPATH=. python -m src.pipeline
```

**Pipeline stages:**
1. `Data Ingestion` — generates synthetic 2-year hourly dataset
2. `EDA` — saves 8 analysis plots to `artifacts/plots/eda/`
3. `Preprocessing` — null imputation, IQR outlier treatment, scaling
4. `Feature Engineering` — 50+ features including lags, rolling stats, cyclical
5. `Model Training` — 4–6 ML models trained and compared
6. `Model Selection` — best model saved as `artifacts/models/best_model.pkl`
7. `Anomaly Detector` — ensemble detector fitted on training residuals
8. `DB Persistence` — metrics saved to PostgreSQL/SQLite

### With LSTM Deep Learning

```bash
PYTHONPATH=. python -m src.pipeline --lstm
# or
make train-lstm
```

### Pipeline Options

```bash
python -m src.pipeline --help

Options:
  --no-eda        Skip EDA plots (faster)
  --lstm          Train LSTM model
  --no-cv         Skip cross-validation
  --data PATH     Use custom data file
```

---

## 🌐 API Usage

### Start the API Server

```bash
make api
# API docs: http://localhost:8000/docs
# ReDoc:    http://localhost:8000/redoc
```

### Endpoints

#### `POST /predict` — Demand Forecast

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2023-06-15T14:00:00",
    "features": {
      "demand_lag_1h": 850.0,
      "demand_lag_24h": 820.0,
      "demand_lag_168h": 830.0,
      "demand_roll_mean_24h": 840.0,
      "demand_roll_std_24h": 45.0,
      "hour": 14,
      "day_of_week": 3,
      "month": 6,
      "is_weekend": 0,
      "is_peak_hour": 0,
      "hour_sin": 0.866,
      "hour_cos": -0.5
    }
  }'
```

**Response:**
```json
{
  "request_id": "a3f1b2c4",
  "timestamp": "2023-06-15T14:00:00",
  "predicted_demand": 847.32,
  "model_name": "lightgbm",
  "model_version": "1.0.0",
  "latency_ms": 12.4,
  "units": "kWh"
}
```

#### `POST /detect-anomaly` — Anomaly Detection

```bash
curl -X POST http://localhost:8000/detect-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2023-06-15T14:00:00",
    "actual_demand": 2500.0,
    "predicted_demand": 850.0,
    "method": "ensemble"
  }'
```

**Response:**
```json
{
  "timestamp": "2023-06-15 14:00:00",
  "actual_demand": 2500.0,
  "predicted_demand": 850.0,
  "anomaly_score": 0.823,
  "is_anomaly": true,
  "severity": "high",
  "method": "ensemble",
  "message": "⚠️ Anomaly detected (severity=high)"
}
```

#### `POST /predict/batch` — Batch Forecasting

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"timestamp": "2023-06-15T14:00:00", "features": {"demand_lag_1h": 850, ...}},
      {"timestamp": "2023-06-15T15:00:00", "features": {"demand_lag_1h": 847, ...}}
    ]
  }'
```

#### `GET /metrics` — Model Performance

```bash
curl http://localhost:8000/metrics
```

```json
{
  "model_name": "lightgbm",
  "model_version": "1.0.0",
  "test_rmse": 42.31,
  "test_mae": 28.15,
  "test_mape": 3.72,
  "test_r2": 0.9834,
  "n_train_samples": 12264,
  "n_test_samples": 2628,
  "last_trained_at": "2024-01-15T10:30:00"
}
```

#### `GET /health` — Service Health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "db_connected": true,
  "uptime_seconds": 3612.5,
  "timestamp": "2024-01-15T10:45:00Z"
}
```

---

## 📊 Dashboard

```bash
make dashboard
# → http://localhost:8501
```

Tabs:
- **📈 Forecast** — Time series with anomaly overlays + heatmap
- **🚨 Anomalies** — Severity breakdown, timeline, records table
- **📊 Model Metrics** — Side-by-side model comparison chart
- **🔍 Live Predict** — Interactive prediction + anomaly check form

---

## 🤖 Model Explanation

### Feature Engineering

| Feature Group | Examples | Purpose |
|---|---|---|
| Lag features | `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h` | Capture autocorrelation |
| Rolling stats | `demand_roll_mean_24h`, `demand_roll_std_24h` | Local trend & volatility |
| Calendar | `hour`, `day_of_week`, `month`, `is_weekend` | Seasonality encoding |
| Cyclical | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Continuous periodic features |
| Event flags | `is_peak_hour`, `is_holiday`, `is_night` | Domain knowledge |

### Models Trained

| Model | Type | Key Strength |
|---|---|---|
| Linear Regression | Baseline | Interpretable |
| Ridge | Baseline | L2 regularised |
| Lasso | Baseline | Feature selection |
| Random Forest | Ensemble | Robust, non-linear |
| XGBoost | Gradient Boosting | High accuracy |
| LightGBM | Gradient Boosting | Fast, memory-efficient |
| LSTM | Deep Learning | Long-range temporal patterns |

### LSTM Architecture

```
Input (seq_length=168, 1 feature)
    → LSTM(128) + BatchNorm
    → Dropout(0.2)
    → LSTM(64) + BatchNorm
    → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, linear)
Loss: Huber (robust to outliers)
Optimizer: Adam (lr=0.001)
Early Stopping: patience=10
```

---

## 🚨 Anomaly Detection

Three complementary detectors combine into an **Ensemble** (majority vote):

| Method | Approach | Use Case |
|---|---|---|
| **Isolation Forest** | Unsupervised tree-based isolation | Novel patterns, global outliers |
| **Z-Score** | Rolling statistical deviation | Sudden spikes/drops |
| **Residual-based** | Model prediction error | Context-aware anomalies |
| **Ensemble** | 2/3 majority vote + avg score | Best overall precision |

**Severity Levels** (based on anomaly score):
- `normal` — score < 0.3
- `low` — 0.3 ≤ score < 0.6
- `medium` — 0.6 ≤ score < 0.8
- `high` — score ≥ 0.8

---

## 🐳 Docker Deployment

### Start Full Stack

```bash
# Build images
make docker-build

# Start database + API + dashboard
make docker-up

# Run training in container
make docker-train

# View logs
make docker-logs

# Stop all services
make docker-down
```

### Service URLs (Docker)

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| pgAdmin | http://localhost:5050 |

---

## 🧪 Testing

```bash
# All tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests (API)
make test-integration

# With coverage report
make test-coverage
# → Open artifacts/coverage/index.html
```

Test coverage:
- `tests/unit/test_preprocessing.py` — Cleaner, temporal split
- `tests/unit/test_features.py` — Feature engineering
- `tests/unit/test_models.py` — All model classes + metrics
- `tests/unit/test_anomaly.py` — All 3 detectors + ensemble
- `tests/integration/test_api.py` — All FastAPI endpoints

---

## 📈 Monitoring

### Latency Tracking

Every prediction request logs latency. The `LatencyTracker` tracks:
- Mean, p50, p95, p99 latency
- SLA breach percentage (default: 500ms SLA)

### Data Drift Detection

```python
from src.utils.monitoring import detect_drift

reports = detect_drift(train_data, production_data)
for r in reports:
    print(f"{r.feature}: PSI={r.psi:.3f} drift_level={r.drift_level}")
```

Drift metrics:
- **KS Test**: p-value < 0.05 → statistically significant distribution shift
- **PSI**: < 0.1 stable | 0.1–0.2 low | > 0.2 significant drift

---

## ⚙️ Configuration Reference

All settings are in `config/config.yaml`. Key sections:

```yaml
data:
  frequency: "1H"         # Time series resolution
  train_ratio: 0.70       # Train/val/test split

preprocessing:
  missing_value_strategy: "interpolate"   # interpolate | ffill | bfill
  outlier_method: "iqr"                   # iqr | zscore | both
  iqr_multiplier: 1.5

models:
  deep_learning:
    lstm:
      sequence_length: 168   # 1 week lookback window
      epochs: 50
      lstm_units: [128, 64]

anomaly_detection:
  isolation_forest:
    contamination: 0.05      # Expected anomaly rate
  severity_thresholds:
    low: 0.3
    medium: 0.6
    high: 0.8
```

Override any setting with environment variables using `${VAR_NAME:-default}` syntax.

---

## 📄 License

MIT — built for educational and production use.
