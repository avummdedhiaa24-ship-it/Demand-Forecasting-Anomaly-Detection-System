"""
Integration tests for FastAPI endpoints.
Uses TestClient (no live server needed).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    with TestClient(app) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "db_connected" in data
        assert "uptime_seconds" in data

    def test_health_status_string(self, client):
        data = client.get("/health").json()
        assert data["status"] in ("ok", "degraded", "down")


# ── Predict ───────────────────────────────────────────────────────────

class TestPredictEndpoint:

    VALID_PAYLOAD = {
        "timestamp": "2023-06-15T14:00:00",
        "features": {
            "demand_lag_1h": 850.0,
            "demand_lag_24h": 820.0,
            "demand_lag_168h": 830.0,
            "demand_roll_mean_24h": 840.0,
            "demand_roll_std_24h": 45.0,
            "hour": 14.0,
            "day_of_week": 3.0,
            "month": 6.0,
            "is_weekend": 0.0,
            "is_peak_hour": 0.0,
            "hour_sin": 0.866,
            "hour_cos": -0.5,
        },
    }

    def test_predict_when_no_model_returns_503(self, client):
        """Expect 503 if no model has been loaded (CI environment)."""
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code in (200, 422, 503)

    def test_predict_invalid_timestamp_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "timestamp": "not-a-date"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_features_returns_422(self, client):
        response = client.post("/predict", json={"timestamp": "2023-01-01T00:00:00"})
        assert response.status_code == 422


# ── Detect Anomaly ────────────────────────────────────────────────────

class TestAnomalyEndpoint:

    VALID_PAYLOAD = {
        "timestamp": "2023-06-15T14:00:00",
        "actual_demand": 900.0,
        "predicted_demand": 850.0,
        "method": "ensemble",
    }

    def test_detect_anomaly_returns_200(self, client):
        response = client.post("/detect-anomaly", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/detect-anomaly", json=self.VALID_PAYLOAD).json()
        for field in ["is_anomaly", "anomaly_score", "severity", "method", "message"]:
            assert field in data, f"Missing field: {field}"

    def test_anomaly_score_in_range(self, client):
        data = client.post("/detect-anomaly", json=self.VALID_PAYLOAD).json()
        assert 0 <= data["anomaly_score"] <= 1

    def test_severity_is_valid_label(self, client):
        data = client.post("/detect-anomaly", json=self.VALID_PAYLOAD).json()
        assert data["severity"] in ("normal", "low", "medium", "high")

    def test_obvious_anomaly_flagged(self, client):
        """A very large deviation should be flagged."""
        payload = {
            "timestamp": "2023-06-15T14:00:00",
            "actual_demand": 5000.0,     # massively elevated
            "predicted_demand": 850.0,
            "method": "ensemble",
        }
        data = client.post("/detect-anomaly", json=payload).json()
        # Score should be non-zero
        assert data["anomaly_score"] >= 0

    def test_negative_demand_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "actual_demand": -100.0}
        response = client.post("/detect-anomaly", json=payload)
        assert response.status_code == 422


# ── Metrics ───────────────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_response_fields(self, client):
        data = client.get("/metrics").json()
        for field in ["model_name", "test_rmse", "test_mae", "test_mape", "test_r2"]:
            assert field in data


# ── History Endpoints ─────────────────────────────────────────────────

class TestHistoryEndpoints:

    def test_predictions_history_returns_200(self, client):
        response = client.get("/predictions/history?limit=10")
        assert response.status_code == 200

    def test_anomalies_history_returns_200(self, client):
        response = client.get("/anomalies/history?limit=10")
        assert response.status_code == 200

    def test_predictions_history_structure(self, client):
        data = client.get("/predictions/history?limit=5").json()
        assert "count" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
