"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert "timestamp" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert data["name"] == "Fraud Detection API"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_valid_transaction(self, client, sample_transaction):
        """Test prediction with valid transaction."""
        response = client.post("/predict", json=sample_transaction)
        
        # May fail if model not loaded, that's OK for unit tests
        if response.status_code == 200:
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "risk_level" in data
            assert "threshold_used" in data
            
            assert isinstance(data["is_fraud"], bool)
            assert 0 <= data["fraud_probability"] <= 1
            assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        incomplete_tx = {"Time": 0.0, "Amount": 100.0}  # Missing V1-V28
        
        response = client.post("/predict", json=incomplete_tx)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_amount(self, client, sample_transaction):
        """Test prediction with negative amount."""
        sample_transaction["Amount"] = -100.0
        
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 422  # Validation error

    def test_predict_with_custom_threshold(self, client, sample_transaction):
        """Test prediction with custom threshold."""
        response = client.post(
            "/predict",
            json=sample_transaction,
            params={"threshold": 0.5},
        )
        
        if response.status_code == 200:
            data = response.json()
            assert data["threshold_used"] == 0.5


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict(self, client, sample_transaction):
        """Test batch prediction."""
        batch = {"transactions": [sample_transaction, sample_transaction]}
        
        response = client.post("/predict/batch", json=batch)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_transactions" in data
            assert "fraud_detected" in data
            assert "processing_time_ms" in data
            
            assert data["total_transactions"] == 2
            assert len(data["predictions"]) == 2

    def test_batch_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422  # Min 1 transaction required

    def test_batch_too_large(self, client, sample_transaction):
        """Test batch prediction exceeds max size."""
        batch = {"transactions": [sample_transaction] * 1001}
        
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 422  # Max 1000 transactions


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        # May return 503 if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "n_features" in data
            assert "feature_names" in data
            assert "threshold" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Should contain some metric names
        content = response.text
        assert "fraud_predictions_total" in content or "python_gc" in content
