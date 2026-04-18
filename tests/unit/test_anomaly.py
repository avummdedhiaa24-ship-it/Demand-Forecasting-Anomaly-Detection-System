"""
Unit tests for anomaly detection module.
"""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.detector import (
    IsolationForestDetector,
    ZScoreDetector,
    ResidualDetector,
    EnsembleDetector,
    results_to_dataframe,
    _score_to_severity,
    AnomalyResult,
)


@pytest.fixture
def normal_series():
    """Normal demand series without anomalies."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.Series(800 + rng.normal(0, 30, n))


@pytest.fixture
def anomalous_series(normal_series):
    """Series with injected anomalies."""
    s = normal_series.copy()
    s.iloc[50] = 2500.0    # spike
    s.iloc[150] = -200.0   # drop (negative)
    s.iloc[250] = 1800.0   # moderate spike
    return s


@pytest.fixture
def feature_df(normal_series):
    """Feature DataFrame for Isolation Forest."""
    rng = np.random.default_rng(0)
    n = len(normal_series)
    return pd.DataFrame({
        "demand": normal_series.values,
        "demand_lag_1h": normal_series.shift(1).fillna(800).values,
        "hour": rng.integers(0, 24, n).astype(float),
        "is_weekend": rng.integers(0, 2, n).astype(float),
    })


# ── Z-Score ───────────────────────────────────────────────────────────

class TestZScoreDetector:

    def test_fit_sets_stats(self, normal_series):
        det = ZScoreDetector()
        det.fit(normal_series)
        assert det._global_mean > 0
        assert det._global_std > 0

    def test_detects_spike(self, normal_series, anomalous_series):
        det = ZScoreDetector(threshold=3.0)
        det.fit(normal_series)
        results = det.predict(anomalous_series)
        scores = {i: r.anomaly_score for i, r in enumerate(results) if r.is_anomaly}
        assert 50 in scores  # spike should be detected

    def test_normal_points_not_flagged(self, normal_series):
        det = ZScoreDetector(threshold=5.0)
        det.fit(normal_series)
        results = det.predict(normal_series)
        n_anomalies = sum(r.is_anomaly for r in results)
        # Very few false positives with high threshold
        assert n_anomalies < len(normal_series) * 0.02

    def test_result_structure(self, normal_series):
        det = ZScoreDetector()
        det.fit(normal_series)
        results = det.predict(normal_series)
        assert len(results) == len(normal_series)
        for r in results[:5]:
            assert 0 <= r.anomaly_score <= 1
            assert r.method == "zscore"
            assert r.severity in {"normal", "low", "medium", "high"}


# ── Isolation Forest ──────────────────────────────────────────────────

class TestIsolationForestDetector:

    def test_fit_and_predict(self, feature_df):
        det = IsolationForestDetector()
        det.fit(feature_df)
        results = det.predict(feature_df)
        assert len(results) == len(feature_df)

    def test_scores_in_range(self, feature_df):
        det = IsolationForestDetector()
        det.fit(feature_df)
        results = det.predict(feature_df)
        for r in results:
            assert 0 <= r.anomaly_score <= 1

    def test_not_fitted_raises(self, feature_df):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            det.predict(feature_df)


# ── Residual Detector ─────────────────────────────────────────────────

class TestResidualDetector:

    def test_fit_with_residuals(self, normal_series):
        y_true = normal_series.values
        y_pred = y_true + np.random.normal(0, 20, len(y_true))
        det = ResidualDetector(threshold_multiplier=3.0)
        det.fit(y_true, y_pred)
        assert det._residual_std > 0

    def test_large_residual_is_anomaly(self, normal_series):
        y_true = normal_series.values
        y_pred = y_true.copy()
        y_pred[100] += 1000.0  # huge residual
        det = ResidualDetector(threshold_multiplier=3.0)
        det.fit(y_true, y_pred)
        results = det.predict(y_true, y_pred)
        assert results[100].is_anomaly

    def test_method_label(self, normal_series):
        y = normal_series.values
        det = ResidualDetector()
        det.fit(y, y)
        results = det.predict(y, y)
        assert all(r.method == "residual" for r in results)


# ── Severity Mapping ──────────────────────────────────────────────────

class TestSeverityMapping:

    def test_low_score_is_normal(self):
        assert _score_to_severity(0.1) == "normal"

    def test_medium_score(self):
        assert _score_to_severity(0.65) == "medium"

    def test_high_score(self):
        assert _score_to_severity(0.9) == "high"

    def test_boundary_values(self):
        assert _score_to_severity(0.0) == "normal"
        assert _score_to_severity(1.0) == "high"


# ── Results DataFrame ─────────────────────────────────────────────────

class TestResultsToDataframe:

    def test_converts_correctly(self, normal_series):
        det = ZScoreDetector()
        det.fit(normal_series)
        results = det.predict(normal_series)
        df = results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert "is_anomaly" in df.columns
        assert "anomaly_score" in df.columns
        assert "severity" in df.columns
        assert "method" in df.columns
        assert len(df) == len(results)
