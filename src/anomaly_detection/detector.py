"""
Anomaly Detection Module
========================
Three complementary anomaly detection methods:
  1. Isolation Forest (unsupervised, model-free)
  2. Z-score (statistical)
  3. Residual-based (model-aware: flags when prediction error is too large)

Each method returns:
  - anomaly_score: float in [0, 1]
  - is_anomaly: bool
  - severity: "normal" | "low" | "medium" | "high"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats

from src.utils.config import cfg
from src.utils.helpers import save_artifact, load_artifact
from src.utils.logger import get_logger, log_execution_time

log = get_logger(__name__)

AD_CFG = cfg.anomaly_detection


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """Result for a single data point."""
    timestamp: Optional[str]
    actual: float
    predicted: Optional[float]
    anomaly_score: float          # [0, 1] — higher = more anomalous
    is_anomaly: bool
    severity: str                 # normal | low | medium | high
    method: str                   # isolation_forest | zscore | residual | ensemble


def _score_to_severity(score: float) -> str:
    """Map anomaly score to human-readable severity label."""
    thresholds = AD_CFG.severity_thresholds
    if score >= thresholds.high:
        return "high"
    elif score >= thresholds.medium:
        return "medium"
    elif score >= thresholds.low:
        return "low"
    return "normal"


# ── Isolation Forest Detector ─────────────────────────────────────────

class IsolationForestDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.
    Works directly on demand values + engineered features.
    """

    def __init__(self) -> None:
        if_cfg = AD_CFG.isolation_forest
        self.model = IsolationForest(
            n_estimators=if_cfg.n_estimators,
            contamination=if_cfg.contamination,
            max_samples=if_cfg.max_samples,
            random_state=if_cfg.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    @log_execution_time
    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        """Fit on training data."""
        log.info("Fitting Isolation Forest on {} samples", len(X))
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies in X. Returns one AnomalyResult per row."""
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector not fitted.")

        raw_scores = self.model.decision_function(X)       # [-0.5, 0.5]
        labels = self.model.predict(X)                     # -1=anomaly, 1=normal

        # Normalise scores to [0, 1] — higher = more anomalous
        normalised = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-8
        )

        results = []
        for i, (score, label) in enumerate(zip(normalised, labels)):
            is_anom = label == -1
            results.append(AnomalyResult(
                timestamp=None,
                actual=float(X["demand"].iloc[i]) if "demand" in X.columns else float("nan"),
                predicted=None,
                anomaly_score=float(score),
                is_anomaly=bool(is_anom),
                severity=_score_to_severity(float(score)) if is_anom else "normal",
                method="isolation_forest",
            ))
        n_anomalies = sum(r.is_anomaly for r in results)
        log.info(
            "Isolation Forest — {}/{} anomalies ({:.1%})",
            n_anomalies, len(results), n_anomalies / len(results),
        )
        return results

    def save(self, path: str) -> None:
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        return load_artifact(path)


# ── Z-Score Detector ─────────────────────────────────────────────────

class ZScoreDetector:
    """
    Statistical anomaly detection using rolling Z-score.
    Identifies points that deviate more than `threshold` std devs
    from a rolling window mean.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        window: int = 168,         # 1 week of hourly data
    ) -> None:
        self.threshold = threshold or AD_CFG.zscore.threshold
        self.window = window
        self._global_mean: float = 0.0
        self._global_std: float = 1.0

    @log_execution_time
    def fit(self, series: pd.Series) -> "ZScoreDetector":
        """Compute global statistics on training data."""
        self._global_mean = float(series.mean())
        self._global_std = float(series.std()) + 1e-8
        log.info(
            "ZScoreDetector fitted — μ={:.2f}, σ={:.2f}, threshold={}",
            self._global_mean, self._global_std, self.threshold,
        )
        return self

    def predict(
        self,
        series: pd.Series,
        timestamps: Optional[pd.Series] = None,
    ) -> List[AnomalyResult]:
        """Detect anomalies using rolling Z-score."""
        # Use rolling stats for local context
        rolling_mean = series.rolling(self.window, min_periods=1).mean()
        rolling_std = series.rolling(self.window, min_periods=1).std().fillna(self._global_std)

        z_scores = np.abs((series.values - rolling_mean.values) / (rolling_std.values + 1e-8))

        # Normalise to [0, 1]
        z_max = max(z_scores.max(), self.threshold + 1e-8)
        normalised_scores = np.clip(z_scores / z_max, 0, 1)

        results = []
        for i, (val, z, score) in enumerate(zip(series.values, z_scores, normalised_scores)):
            is_anom = bool(z > self.threshold)
            ts = str(timestamps.iloc[i]) if timestamps is not None else None
            results.append(AnomalyResult(
                timestamp=ts,
                actual=float(val),
                predicted=None,
                anomaly_score=float(score),
                is_anomaly=is_anom,
                severity=_score_to_severity(float(score)) if is_anom else "normal",
                method="zscore",
            ))

        n_anomalies = sum(r.is_anomaly for r in results)
        log.info("Z-score — {}/{} anomalies", n_anomalies, len(results))
        return results


# ── Residual-Based Detector ───────────────────────────────────────────

class ResidualDetector:
    """
    Model-aware anomaly detection.
    Flags observations where |actual - predicted| > k × std(residuals_train).
    """

    def __init__(self, threshold_multiplier: Optional[float] = None) -> None:
        self.k = threshold_multiplier or AD_CFG.residual_based.threshold_multiplier
        self._residual_std: float = 1.0
        self._residual_mean: float = 0.0
        self.is_fitted = False

    @log_execution_time
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ResidualDetector":
        """Calibrate on training residuals."""
        residuals = y_true - y_pred
        self._residual_mean = float(residuals.mean())
        self._residual_std = float(residuals.std()) + 1e-8
        self.is_fitted = True
        log.info(
            "ResidualDetector fitted — residual μ={:.2f}, σ={:.2f}, k={}",
            self._residual_mean, self._residual_std, self.k,
        )
        return self

    def predict(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.Series] = None,
    ) -> List[AnomalyResult]:
        """Detect anomalies in new actual vs predicted pairs."""
        if not self.is_fitted:
            raise RuntimeError("ResidualDetector not fitted.")

        residuals = y_true - y_pred
        threshold = self.k * self._residual_std

        # Anomaly score: normalised absolute residual
        abs_res = np.abs(residuals)
        scores = np.clip(abs_res / (self.k * self._residual_std + 1e-8), 0, 1)

        results = []
        for i, (actual, pred, res, score) in enumerate(
            zip(y_true, y_pred, residuals, scores)
        ):
            is_anom = bool(abs(res) > threshold)
            ts = str(timestamps.iloc[i]) if timestamps is not None else None
            results.append(AnomalyResult(
                timestamp=ts,
                actual=float(actual),
                predicted=float(pred),
                anomaly_score=float(score),
                is_anomaly=is_anom,
                severity=_score_to_severity(float(score)) if is_anom else "normal",
                method="residual",
            ))

        n_anomalies = sum(r.is_anomaly for r in results)
        log.info(
            "Residual detector — {}/{} anomalies (threshold={:.2f})",
            n_anomalies, len(results), threshold,
        )
        return results


# ── Ensemble Detector ─────────────────────────────────────────────────

class EnsembleDetector:
    """
    Combines all three detectors via majority voting + averaged scores.
    A point is labelled anomaly if ≥ 2/3 detectors agree.
    """

    def __init__(self) -> None:
        self.if_detector = IsolationForestDetector()
        self.zscore_detector = ZScoreDetector()
        self.residual_detector = ResidualDetector()
        self.is_fitted = False

    @log_execution_time
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_pred_train: np.ndarray,
    ) -> "EnsembleDetector":
        """Fit all sub-detectors on training data."""
        self.if_detector.fit(X_train)
        self.zscore_detector.fit(y_train)
        self.residual_detector.fit(y_train.values, y_pred_train)
        self.is_fitted = True
        log.info("EnsembleDetector fitted on {} samples", len(X_train))
        return self

    def predict(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.Series] = None,
    ) -> List[AnomalyResult]:
        """Run ensemble detection and return combined results."""
        if not self.is_fitted:
            raise RuntimeError("EnsembleDetector not fitted.")

        demand_series = pd.Series(y_true)

        if_results = self.if_detector.predict(X)
        zs_results = self.zscore_detector.predict(demand_series, timestamps)
        res_results = self.residual_detector.predict(y_true, y_pred, timestamps)

        combined = []
        for i, (r_if, r_zs, r_res) in enumerate(
            zip(if_results, zs_results, res_results)
        ):
            votes = sum([r_if.is_anomaly, r_zs.is_anomaly, r_res.is_anomaly])
            is_anom = votes >= 2  # majority vote
            avg_score = (r_if.anomaly_score + r_zs.anomaly_score + r_res.anomaly_score) / 3

            ts = str(timestamps.iloc[i]) if timestamps is not None else None
            combined.append(AnomalyResult(
                timestamp=ts,
                actual=float(y_true[i]),
                predicted=float(y_pred[i]),
                anomaly_score=float(avg_score),
                is_anomaly=bool(is_anom),
                severity=_score_to_severity(float(avg_score)) if is_anom else "normal",
                method="ensemble",
            ))

        n_anom = sum(r.is_anomaly for r in combined)
        log.info(
            "Ensemble — {}/{} anomalies ({:.1%})",
            n_anom, len(combined), n_anom / len(combined),
        )
        return combined

    def save(self, path: str) -> None:
        save_artifact(self, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleDetector":
        return load_artifact(path)


# ── Results → DataFrame ───────────────────────────────────────────────

def results_to_dataframe(results: List[AnomalyResult]) -> pd.DataFrame:
    """Convert list of AnomalyResult to a pandas DataFrame."""
    return pd.DataFrame([
        {
            "timestamp": r.timestamp,
            "actual": r.actual,
            "predicted": r.predicted,
            "anomaly_score": r.anomaly_score,
            "is_anomaly": r.is_anomaly,
            "severity": r.severity,
            "method": r.method,
        }
        for r in results
    ])
