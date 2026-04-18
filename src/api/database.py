"""
Database Layer
==============
PostgreSQL connection via SQLAlchemy 2.0 ORM.
Tables: time_series_data, predictions, anomalies.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    String, Text, create_engine, text, Index,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Connection ────────────────────────────────────────────────────────

def _build_db_url() -> str:
    db = cfg.database
    host = os.environ.get("DB_HOST", db.host)
    port = os.environ.get("DB_PORT", str(db.port))
    name = os.environ.get("DB_NAME", db.name)
    user = os.environ.get("DB_USER", db.user)
    password = os.environ.get("DB_PASSWORD", db.password)
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def _build_sqlite_url() -> str:
    """Fallback SQLite URL for local development without PostgreSQL."""
    return "sqlite:///./demand_forecast.db"


def create_db_engine(sqlite_fallback: bool = True):
    """Create SQLAlchemy engine. Falls back to SQLite if PostgreSQL unavailable."""
    db_cfg = cfg.database
    url = _build_db_url()

    try:
        engine = create_engine(
            url,
            pool_size=db_cfg.pool_size,
            max_overflow=db_cfg.max_overflow,
            pool_timeout=db_cfg.pool_timeout,
            echo=db_cfg.echo,
            pool_pre_ping=True,
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("PostgreSQL connected at {}", url.split("@")[-1])
        return engine
    except Exception as exc:
        if sqlite_fallback:
            log.warning(
                "PostgreSQL unavailable ({}). Falling back to SQLite.",
                str(exc)[:80],
            )
            return create_engine(
                _build_sqlite_url(),
                connect_args={"check_same_thread": False},
                echo=False,
            )
        raise


# ── ORM Models ────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class TimeSeriesData(Base):
    """
    Raw / cleaned time-series ingestion records.
    """
    __tablename__ = "time_series_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, unique=True)
    demand = Column(Float, nullable=False)
    source = Column(String(64), default="synthetic")
    is_processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_ts_data_timestamp_source", "timestamp", "source"),
    )

    def __repr__(self) -> str:
        return f"<TimeSeriesData ts={self.timestamp} demand={self.demand:.2f}>"


class Prediction(Base):
    """
    Model prediction records — one row per prediction request.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    predicted_demand = Column(Float, nullable=False)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), default="1.0.0")
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<Prediction ts={self.timestamp} "
            f"pred={self.predicted_demand:.2f} model={self.model_name}>"
        )


class Anomaly(Base):
    """
    Detected anomaly records.
    """
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    actual_demand = Column(Float, nullable=False)
    predicted_demand = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    severity = Column(String(16), nullable=False, default="normal")
    method = Column(String(32), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<Anomaly ts={self.timestamp} "
            f"score={self.anomaly_score:.3f} severity={self.severity}>"
        )


class ModelMetrics(Base):
    """
    Snapshot of model evaluation metrics after each training run.
    """
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False)
    rmse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    r2 = Column(Float)
    n_train_samples = Column(Integer)
    n_test_samples = Column(Integer)
    training_duration_s = Column(Float)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Session Factory ───────────────────────────────────────────────────

engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    log.info("Database tables initialised")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Repository Helpers ────────────────────────────────────────────────

class PredictionRepository:
    """CRUD for the predictions table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def insert(self, record: Prediction) -> Prediction:
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_recent(self, limit: int = 100) -> list[Prediction]:
        return (
            self.db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_by_request_id(self, request_id: str) -> Optional[Prediction]:
        return (
            self.db.query(Prediction)
            .filter(Prediction.request_id == request_id)
            .first()
        )


class AnomalyRepository:
    """CRUD for the anomalies table."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def insert(self, record: Anomaly) -> Anomaly:
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_anomalies(
        self,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[Anomaly]:
        q = self.db.query(Anomaly).filter(Anomaly.is_anomaly == True)
        if severity:
            q = q.filter(Anomaly.severity == severity)
        return q.order_by(Anomaly.timestamp.desc()).limit(limit).all()
