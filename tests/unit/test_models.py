"""
Unit tests for forecasting models.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_models import (
    LinearRegressionForecaster,
    LassoForecaster,
    RidgeForecaster,
)
from src.models.advanced_models import RandomForestForecaster
from src.evaluation.metrics import rmse, mae, mape, r2, compute_all_metrics


@pytest.fixture
def simple_dataset():
    """Simple regression dataset for model tests."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({
        "demand_lag_1h": 800 + rng.normal(0, 50, n),
        "demand_lag_24h": 800 + rng.normal(0, 40, n),
        "hour": rng.integers(0, 24, n).astype(float),
        "day_of_week": rng.integers(0, 7, n).astype(float),
        "is_weekend": rng.integers(0, 2, n).astype(float),
        "hour_sin": np.sin(2 * np.pi * rng.integers(0, 24, n) / 24),
        "hour_cos": np.cos(2 * np.pi * rng.integers(0, 24, n) / 24),
        "demand_roll_mean_24h": 800 + rng.normal(0, 30, n),
        "demand_roll_std_24h": rng.uniform(10, 60, n),
    })
    y = pd.Series(
        0.7 * X["demand_lag_1h"].values +
        0.2 * X["demand_lag_24h"].values +
        10 * X["hour"].values +
        rng.normal(0, 20, n),
        name="demand",
    )
    y = y.clip(0)
    split = int(0.8 * n)
    return X[:split], y[:split], X[split:], y[split:]


# ── Baseline Model Tests ──────────────────────────────────────────────

class TestBaselineModels:

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionForecaster,
        RidgeForecaster,
        LassoForecaster,
    ])
    def test_fit_returns_self(self, ModelClass, simple_dataset):
        X_tr, y_tr, _, _ = simple_dataset
        model = ModelClass()
        result = model.fit(X_tr, y_tr)
        assert result is model

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionForecaster,
        RidgeForecaster,
        LassoForecaster,
    ])
    def test_predict_shape(self, ModelClass, simple_dataset):
        X_tr, y_tr, X_te, _ = simple_dataset
        model = ModelClass()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)

    @pytest.mark.parametrize("ModelClass", [
        LinearRegressionForecaster,
        RidgeForecaster,
        LassoForecaster,
    ])
    def test_predict_non_negative(self, ModelClass, simple_dataset):
        X_tr, y_tr, X_te, _ = simple_dataset
        model = ModelClass()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert (preds >= 0).all()

    def test_not_fitted_raises(self):
        model = LinearRegressionForecaster()
        X = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_is_fitted_flag(self, simple_dataset):
        X_tr, y_tr, _, _ = simple_dataset
        model = RidgeForecaster()
        assert not model.is_fitted
        model.fit(X_tr, y_tr)
        assert model.is_fitted


# ── Advanced Model Tests ──────────────────────────────────────────────

class TestRandomForest:

    def test_fit_and_predict(self, simple_dataset):
        X_tr, y_tr, X_te, y_te = simple_dataset
        model = RandomForestForecaster()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert (preds >= 0).all()

    def test_feature_importances(self, simple_dataset):
        X_tr, y_tr, _, _ = simple_dataset
        model = RandomForestForecaster()
        model.fit(X_tr, y_tr)
        importances = model.feature_importances
        assert len(importances) == X_tr.shape[1]
        assert importances.sum() == pytest.approx(1.0, abs=1e-5)

    def test_reasonable_rmse(self, simple_dataset):
        X_tr, y_tr, X_te, y_te = simple_dataset
        model = RandomForestForecaster()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        score = rmse(y_te.values, preds)
        assert score < 200   # sanity check on synthetic data


# ── Metrics Tests ─────────────────────────────────────────────────────

class TestMetrics:

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mape_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 220.0])
        # (10/100 + 20/200) / 2 = (0.1 + 0.1) / 2 = 10%
        result = mape(y_true, y_pred)
        assert result == pytest.approx(10.0, abs=0.1)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_r2_zero_for_mean_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred_mean = np.full_like(y, y.mean())
        score = r2(y, pred_mean)
        assert abs(score) < 0.01

    def test_compute_all_metrics_keys(self):
        y = np.array([100.0, 200.0, 300.0])
        p = np.array([105.0, 195.0, 310.0])
        metrics = compute_all_metrics(y, p, prefix="test")
        expected_keys = {"test_rmse", "test_mae", "test_mape", "test_r2", "test_smape"}
        assert expected_keys.issubset(set(metrics.keys()))

    def test_metrics_non_negative(self):
        y = np.array([100.0, 200.0, 300.0])
        p = np.array([90.0, 210.0, 305.0])
        metrics = compute_all_metrics(y, p)
        for key in ["rmse", "mae", "mape"]:
            assert metrics[key] >= 0
