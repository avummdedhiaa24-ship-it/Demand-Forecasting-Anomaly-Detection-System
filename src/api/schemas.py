"""
API Pydantic Schemas
====================
Request and response models for all FastAPI endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Predict ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input for /predict endpoint."""
    timestamp: datetime = Field(..., description="Target forecast timestamp (ISO-8601)")
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Feature dict — must contain at minimum recent lag values. "
            "Keys: demand_lag_1h, demand_lag_24h, demand_lag_168h, "
            "demand_roll_mean_24h, hour, day_of_week, month, etc."
        ),
        example={
            "demand_lag_1h": 850.5,
            "demand_lag_24h": 820.3,
            "demand_lag_168h": 830.0,
            "demand_roll_mean_24h": 840.2,
            "demand_roll_std_24h": 45.1,
            "hour": 14,
            "day_of_week": 1,
            "month": 6,
            "is_weekend": 0,
            "is_peak_hour": 0,
            "hour_sin": 0.866,
            "hour_cos": -0.5,
        },
    )
    model_name: Optional[str] = Field(
        None,
        description="Which model to use (default: best model from training).",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class PredictResponse(BaseModel):
    """Output from /predict endpoint."""
    request_id: str
    timestamp: str
    predicted_demand: float = Field(..., description="Forecast in kWh")
    model_name: str
    model_version: str = "1.0.0"
    latency_ms: float
    units: str = "kWh"


# ── Detect Anomaly ────────────────────────────────────────────────────

class AnomalyDetectRequest(BaseModel):
    """Input for /detect-anomaly endpoint."""
    timestamp: datetime
    actual_demand: float = Field(..., ge=0, description="Observed demand in kWh")
    predicted_demand: Optional[float] = Field(
        None, description="Model-predicted demand (if available)"
    )
    features: Optional[Dict[str, float]] = Field(
        None, description="Feature dict for Isolation Forest scoring"
    )
    method: str = Field(
        "ensemble",
        description="Detection method: isolation_forest | zscore | residual | ensemble",
    )


class AnomalyDetectResponse(BaseModel):
    """Output from /detect-anomaly endpoint."""
    timestamp: str
    actual_demand: float
    predicted_demand: Optional[float]
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomaly: bool
    severity: str = Field(..., description="normal | low | medium | high")
    method: str
    message: str


# ── Batch Predict ─────────────────────────────────────────────────────

class BatchPredictRequest(BaseModel):
    """Input for /predict/batch endpoint."""
    records: List[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Output from /predict/batch endpoint."""
    count: int
    predictions: List[PredictResponse]
    total_latency_ms: float


# ── Metrics ───────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    """Output from /metrics endpoint."""
    model_name: str
    model_version: str
    test_rmse: float
    test_mae: float
    test_mape: float
    test_r2: float
    n_train_samples: int
    n_test_samples: int
    last_trained_at: Optional[str]


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Output from /health endpoint."""
    status: str                              # ok | degraded | down
    version: str
    model_loaded: bool
    db_connected: bool
    uptime_seconds: float
    timestamp: str


# ── History ───────────────────────────────────────────────────────────

class PredictionHistoryItem(BaseModel):
    id: int
    request_id: str
    timestamp: str
    predicted_demand: float
    model_name: str
    latency_ms: Optional[float]
    created_at: str


class AnomalyHistoryItem(BaseModel):
    id: int
    timestamp: str
    actual_demand: float
    predicted_demand: Optional[float]
    anomaly_score: float
    is_anomaly: bool
    severity: str
    method: str
    created_at: str
