"""
Monitoring & Data Drift Detection
===================================
Detects distribution shift between training and production data.
Logs prediction latency and system health metrics.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

MON_CFG = cfg.monitoring


# ── Latency Tracker ───────────────────────────────────────────────────

class LatencyTracker:
    """
    Rolling window tracker for prediction latency.
    Alerts when p99 latency exceeds SLA threshold.
    """

    def __init__(
        self,
        window: int = 1000,
        sla_ms: Optional[float] = None,
    ) -> None:
        self.window = window
        self.sla_ms = sla_ms or MON_CFG.latency_sla_ms
        self._buffer: Deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        self._buffer.append(latency_ms)
        if latency_ms > self.sla_ms:
            log.warning("⚠️ Latency SLA breach: {:.1f}ms > {}ms", latency_ms, self.sla_ms)

    def summary(self) -> Dict[str, float]:
        if not self._buffer:
            return {}
        arr = np.array(self._buffer)
        return {
            "count": len(arr),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
            "sla_breach_pct": float((arr > self.sla_ms).mean() * 100),
        }


# ── Drift Detection ───────────────────────────────────────────────────

@dataclass
class DriftReport:
    """Results of a drift detection comparison."""
    feature: str
    train_mean: float
    prod_mean: float
    train_std: float
    prod_std: float
    ks_statistic: float
    ks_pvalue: float
    psi: float                  # Population Stability Index
    is_drifted: bool
    drift_level: str            # none | low | medium | high


def _compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """
    Population Stability Index.
    PSI < 0.1 → stable, 0.1–0.2 → low drift, > 0.2 → significant drift.
    """
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / (len(expected) + eps) + eps
    act_pct = act_counts / (len(actual) + eps) + eps

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


def detect_drift(
    train_data: pd.DataFrame,
    prod_data: pd.DataFrame,
    features: Optional[List[str]] = None,
    threshold: Optional[float] = None,
) -> List[DriftReport]:
    """
    Compare production data distribution against training baseline.

    Uses:
      - Kolmogorov-Smirnov test (p-value < 0.05 → drift)
      - Population Stability Index (PSI > 0.1 → drift)

    Args:
        train_data: Training distribution baseline.
        prod_data: Production data window.
        features: Columns to check (defaults to all numeric).
        threshold: PSI threshold for drift alert.

    Returns:
        List of DriftReport for each feature.
    """
    threshold = threshold or MON_CFG.drift_threshold
    features = features or train_data.select_dtypes(include=[np.number]).columns.tolist()

    reports = []
    n_drifted = 0

    for feat in features:
        if feat not in train_data.columns or feat not in prod_data.columns:
            continue

        train_vals = train_data[feat].dropna().values
        prod_vals = prod_data[feat].dropna().values

        if len(train_vals) < 10 or len(prod_vals) < 10:
            continue

        ks_stat, ks_pval = stats.ks_2samp(train_vals, prod_vals)
        psi = _compute_psi(train_vals, prod_vals)

        is_drifted = (ks_pval < 0.05) or (psi > threshold)
        drift_level = _psi_to_level(psi)

        if is_drifted:
            n_drifted += 1
            log.warning(
                "📊 Drift detected in '{}': PSI={:.3f}, KS p={:.4f}",
                feat, psi, ks_pval,
            )

        reports.append(DriftReport(
            feature=feat,
            train_mean=float(train_vals.mean()),
            prod_mean=float(prod_vals.mean()),
            train_std=float(train_vals.std()),
            prod_std=float(prod_vals.std()),
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            psi=float(psi),
            is_drifted=is_drifted,
            drift_level=drift_level,
        ))

    log.info(
        "Drift check: {}/{} features drifted", n_drifted, len(reports)
    )
    return reports


def _psi_to_level(psi: float) -> str:
    if psi < 0.1:
        return "none"
    elif psi < 0.2:
        return "low"
    elif psi < 0.25:
        return "medium"
    return "high"


# ── Global Tracker Singleton ──────────────────────────────────────────

latency_tracker = LatencyTracker()
