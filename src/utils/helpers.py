"""
Shared helper utilities used across the system.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Serialisation ─────────────────────────────────────────────────────

def save_artifact(obj: Any, path: str | Path, method: str = "pickle") -> None:
    """
    Persist a Python object to disk.

    Args:
        obj: Object to serialise.
        path: Destination file path.
        method: 'pickle' or 'json'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if method == "pickle":
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == "json":
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=_json_serialiser)
    else:
        raise ValueError(f"Unknown serialisation method: {method}")

    log.info("Saved artifact → {}", path)


def load_artifact(path: str | Path, method: str = "pickle") -> Any:
    """Load a previously saved artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    if method == "pickle":
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif method == "json":
        with open(path, "r") as f:
            obj = json.load(f)
    else:
        raise ValueError(f"Unknown method: {method}")

    log.info("Loaded artifact ← {}", path)
    return obj


def _json_serialiser(obj: Any) -> Any:
    """JSON serialiser for non-standard types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# ── DataFrame Helpers ─────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str = "DataFrame",
) -> None:
    """Raise ValueError if required columns are missing."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    log.debug("{} validated — shape={}, cols={}", name, df.shape, list(df.columns))


def memory_usage(df: pd.DataFrame) -> str:
    """Return human-readable DataFrame memory usage."""
    bytes_ = df.memory_usage(deep=True).sum()
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint by downcasting numeric types."""
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ── Metrics Helpers ───────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error."""
    naive_mae = np.mean(np.abs(np.diff(y_train, n=seasonality)))
    if naive_mae == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


# ── Hashing ───────────────────────────────────────────────────────────

def hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 hash of a DataFrame."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


# ── Time Utilities ────────────────────────────────────────────────────

def timestamp_now() -> str:
    """ISO-format UTC timestamp string."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def floor_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    """Floor a timestamp to the nearest hour."""
    return ts.floor("h")
