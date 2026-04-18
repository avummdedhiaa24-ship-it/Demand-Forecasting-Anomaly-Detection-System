"""
FastAPI Application
===================
Exposes prediction, anomaly detection, metrics, and health endpoints.
Includes request logging, error handling, and Prometheus metrics.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.api.database import (
    Anomaly, AnomalyRepository, ModelMetrics, Prediction,
    PredictionRepository, get_db, init_db,
)
from src.api.schemas import (
    AnomalyDetectRequest, AnomalyDetectResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, MetricsResponse,
    PredictRequest, PredictResponse,
)
from src.anomaly_detection.detector import (
    AnomalyResult, EnsembleDetector, ZScoreDetector,
    IsolationForestDetector, ResidualDetector,
)
from src.models.base_model import BaseForecaster
from src.utils.config import cfg
from src.utils.helpers import load_artifact, timestamp_now
from src.utils.logger import get_logger, log_async_execution_time

log = get_logger(__name__)

# ── Application State ─────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "model": None,
    "anomaly_detector": None,
    "metrics": {},
    "start_time": time.time(),
    "model_loaded": False,
    "db_connected": False,
}

MODEL_PATH = Path(cfg.api.model_path)
DETECTOR_PATH = Path("artifacts/models/ensemble_detector.pkl")


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    log.info("🚀 Starting Demand Forecasting API v{}", cfg.project.version)

    # Init DB
    try:
        init_db()
        _state["db_connected"] = True
        log.info("✅ Database initialised")
    except Exception as exc:
        log.error("❌ DB init failed: {}", exc)

    # Load model
    try:
        if MODEL_PATH.exists():
            _state["model"] = load_artifact(MODEL_PATH)
            _state["model_loaded"] = True
            log.info("✅ Model loaded from {}", MODEL_PATH)
        else:
            log.warning("⚠️  No trained model found at {}. Train first.", MODEL_PATH)
    except Exception as exc:
        log.error("❌ Model load failed: {}", exc)

    # Load anomaly detector
    try:
        if DETECTOR_PATH.exists():
            _state["anomaly_detector"] = load_artifact(DETECTOR_PATH)
            log.info("✅ Anomaly detector loaded")
    except Exception as exc:
        log.warning("Anomaly detector not loaded: {}", exc)

    yield  # ← Application runs here

    log.info("👋 API shutting down")


# ── App Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Intelligent Demand Forecasting & Anomaly Detection API",
        description=(
            "End-to-end ML system for time-series demand forecasting "
            "and real-time anomaly detection."
        ),
        version=cfg.project.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "{} {} → {} [{:.1f}ms]",
            request.method, request.url.path,
            response.status_code, elapsed_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ── Error Handlers ────────────────────────────────────────────────
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        log.error("Runtime error: {}", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    return app


app = create_app()


# ── Dependency Helpers ────────────────────────────────────────────────

def get_model() -> BaseForecaster:
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the training pipeline first.",
        )
    return model


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Service health and readiness check."""
    uptime = time.time() - _state["start_time"]
    return HealthResponse(
        status="ok" if _state["model_loaded"] else "degraded",
        version=cfg.project.version,
        model_loaded=_state["model_loaded"],
        db_connected=_state["db_connected"],
        uptime_seconds=round(uptime, 2),
        timestamp=timestamp_now(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
@log_async_execution_time
async def predict(
    body: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Generate a demand forecast for a given timestamp.

    Accepts a feature dict built from the feature engineering pipeline.
    Returns predicted demand in kWh.
    """
    model = get_model()
    request_id = str(uuid.uuid4())[:16]
    start = time.perf_counter()

    # Build feature DataFrame
    X = pd.DataFrame([body.features])
    X = X.reindex(
        columns=model.feature_names or list(body.features.keys()),
        fill_value=0,
    ).astype(float)

    try:
        pred = model.predict(X)[0]
    except Exception as exc:
        log.error("Prediction failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    latency_ms = (time.perf_counter() - start) * 1000

    # Persist to DB
    try:
        repo = PredictionRepository(db)
        repo.insert(Prediction(
            request_id=request_id,
            timestamp=body.timestamp,
            predicted_demand=float(pred),
            model_name=model.name,
            latency_ms=latency_ms,
        ))
    except Exception as exc:
        log.warning("Failed to persist prediction: {}", exc)

    return PredictResponse(
        request_id=request_id,
        timestamp=body.timestamp.isoformat(),
        predicted_demand=round(float(pred), 4),
        model_name=model.name,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Forecasting"])
async def predict_batch(
    body: BatchPredictRequest,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """
    Batch demand forecasting for up to 1000 records.
    """
    model = get_model()
    start_total = time.perf_counter()
    predictions = []

    feature_keys = model.feature_names or list(body.records[0].features.keys())
    rows = [r.features for r in body.records]
    X_batch = pd.DataFrame(rows).reindex(columns=feature_keys, fill_value=0).astype(float)

    try:
        preds = model.predict(X_batch)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {exc}",
        )

    total_ms = (time.perf_counter() - start_total) * 1000

    for i, (req, pred) in enumerate(zip(body.records, preds)):
        predictions.append(PredictResponse(
            request_id=f"batch_{i}_{str(uuid.uuid4())[:8]}",
            timestamp=req.timestamp.isoformat(),
            predicted_demand=round(float(pred), 4),
            model_name=model.name,
            latency_ms=round(total_ms / len(body.records), 2),
        ))

    return BatchPredictResponse(
        count=len(predictions),
        predictions=predictions,
        total_latency_ms=round(total_ms, 2),
    )


@app.post("/detect-anomaly", response_model=AnomalyDetectResponse, tags=["Anomaly Detection"])
@log_async_execution_time
async def detect_anomaly(
    body: AnomalyDetectRequest,
    db: Session = Depends(get_db),
) -> AnomalyDetectResponse:
    """
    Detect whether a demand observation is anomalous.

    Supports: ensemble (default), zscore, isolation_forest, residual.
    """
    detector = _state.get("anomaly_detector")
    pred_demand = body.predicted_demand

    # Fallback: use model prediction if predicted_demand not supplied
    if pred_demand is None and _state["model_loaded"]:
        try:
            model = get_model()
            if body.features:
                X = pd.DataFrame([body.features]).astype(float)
                pred_demand = float(model.predict(X)[0])
        except Exception:
            pass

    # Compute anomaly result
    if detector is not None and isinstance(detector, EnsembleDetector) and body.features:
        # Full ensemble
        try:
            features_df = pd.DataFrame([body.features])
            y_true = np.array([body.actual_demand])
            y_pred = np.array([pred_demand or body.actual_demand])
            results = detector.predict(features_df, y_true, y_pred)
            result = results[0]
        except Exception as exc:
            log.warning("Ensemble detector failed, falling back to zscore: {}", exc)
            detector = None
    if detector is None or not isinstance(detector, EnsembleDetector) or not body.features:
        # Fallback: Z-score only
        z = abs(body.actual_demand - (pred_demand or body.actual_demand))
        score = min(z / (body.actual_demand + 1e-8), 1.0)
        is_anom = score > 0.3
        from src.anomaly_detection.detector import _score_to_severity
        result = AnomalyResult(
            timestamp=str(body.timestamp),
            actual=body.actual_demand,
            predicted=pred_demand,
            anomaly_score=float(score),
            is_anomaly=is_anom,
            severity=_score_to_severity(float(score)) if is_anom else "normal",
            method="zscore_fallback",
        )

    # Persist
    try:
        anomaly_repo = AnomalyRepository(db)
        anomaly_repo.insert(Anomaly(
            timestamp=body.timestamp,
            actual_demand=result.actual,
            predicted_demand=result.predicted,
            anomaly_score=result.anomaly_score,
            is_anomaly=result.is_anomaly,
            severity=result.severity,
            method=result.method,
        ))
    except Exception as exc:
        log.warning("Failed to persist anomaly record: {}", exc)

    msg = (
        f"⚠️ Anomaly detected (severity={result.severity})"
        if result.is_anomaly
        else "✅ Normal observation"
    )

    return AnomalyDetectResponse(
        timestamp=str(body.timestamp),
        actual_demand=result.actual,
        predicted_demand=result.predicted,
        anomaly_score=round(result.anomaly_score, 4),
        is_anomaly=result.is_anomaly,
        severity=result.severity,
        method=result.method,
        message=msg,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    """
    Return latest model performance metrics from the database.
    """
    try:
        record = (
            db.query(ModelMetrics)
            .order_by(ModelMetrics.created_at.desc())
            .first()
        )
    except Exception:
        record = None

    if record:
        return MetricsResponse(
            model_name=record.model_name,
            model_version=record.model_version,
            test_rmse=record.rmse or 0.0,
            test_mae=record.mae or 0.0,
            test_mape=record.mape or 0.0,
            test_r2=record.r2 or 0.0,
            n_train_samples=record.n_train_samples or 0,
            n_test_samples=record.n_test_samples or 0,
            last_trained_at=str(record.created_at) if record.created_at else None,
        )

    # Fallback: return in-memory metrics
    m = _state.get("metrics", {})
    model = _state.get("model")
    return MetricsResponse(
        model_name=model.name if model else "none",
        model_version="1.0.0",
        test_rmse=m.get("test_rmse", 0.0),
        test_mae=m.get("test_mae", 0.0),
        test_mape=m.get("test_mape", 0.0),
        test_r2=m.get("test_r2", 0.0),
        n_train_samples=m.get("n_train_samples", 0),
        n_test_samples=m.get("n_test_samples", 0),
        last_trained_at=m.get("last_trained_at"),
    )


@app.get("/predictions/history", tags=["History"])
async def prediction_history(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Return recent prediction history."""
    repo = PredictionRepository(db)
    records = repo.get_recent(limit=limit)
    return {
        "count": len(records),
        "predictions": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "timestamp": str(r.timestamp),
                "predicted_demand": r.predicted_demand,
                "model_name": r.model_name,
                "latency_ms": r.latency_ms,
                "created_at": str(r.created_at),
            }
            for r in records
        ],
    }


@app.get("/anomalies/history", tags=["History"])
async def anomaly_history(
    severity: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Return detected anomalies, optionally filtered by severity."""
    repo = AnomalyRepository(db)
    records = repo.get_anomalies(severity=severity, limit=limit)
    return {
        "count": len(records),
        "anomalies": [
            {
                "id": r.id,
                "timestamp": str(r.timestamp),
                "actual_demand": r.actual_demand,
                "predicted_demand": r.predicted_demand,
                "anomaly_score": r.anomaly_score,
                "is_anomaly": r.is_anomaly,
                "severity": r.severity,
                "method": r.method,
                "created_at": str(r.created_at),
            }
            for r in records
        ],
    }


# ── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        workers=1,
        log_level="info",
    )
