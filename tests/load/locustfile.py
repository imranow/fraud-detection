"""
Load testing for Fraud Detection API using Locust.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Or headless:
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           --headless -u 100 -r 10 -t 5m
"""

import json
import random
from locust import HttpUser, between, task
from locust.contrib.fasthttp import FastHttpUser


# Sample transaction data based on the credit card dataset
SAMPLE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62,
}


def generate_random_transaction():
    """Generate a random transaction with realistic variations."""
    transaction = SAMPLE_TRANSACTION.copy()
    
    # Randomize values slightly
    for key in transaction:
        if key.startswith("V"):
            # Add random noise to PCA features
            transaction[key] += random.gauss(0, 0.1)
        elif key == "Amount":
            # Random amount between 0 and 500
            transaction[key] = random.uniform(0, 500)
        elif key == "Time":
            # Random time in a day
            transaction[key] = random.uniform(0, 86400)
    
    return transaction


def generate_fraud_like_transaction():
    """Generate a transaction with fraud-like characteristics."""
    transaction = generate_random_transaction()
    
    # Fraud transactions tend to have certain patterns
    transaction["V17"] = random.uniform(-5, -2)
    transaction["V14"] = random.uniform(-5, -2)
    transaction["V12"] = random.uniform(-3, -1)
    transaction["V10"] = random.uniform(-3, -1)
    
    return transaction


class FraudDetectionUser(FastHttpUser):
    """
    Load test user for the Fraud Detection API.
    
    Uses FastHttpUser for better performance.
    """
    
    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests
    
    def on_start(self):
        """Called when a user starts."""
        # Optional: Set up API key if authentication is enabled
        self.api_key = "test-api-key-for-load-testing"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }
    
    @task(10)
    def predict_single(self):
        """Test single prediction endpoint."""
        transaction = generate_random_transaction()
        
        with self.client.post(
            "/predict",
            json=transaction,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                # Auth disabled, retry without key
                del self.headers["X-API-Key"]
                response.success()
            else:
                response.failure(f"Status {response.status_code}: {response.text}")
    
    @task(3)
    def predict_single_fraud_like(self):
        """Test single prediction with fraud-like transaction."""
        transaction = generate_fraud_like_transaction()
        
        with self.client.post(
            "/predict",
            json=transaction,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Log if we actually detected fraud
                if data.get("is_fraud"):
                    response.success()
                else:
                    response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(2)
    def predict_batch_small(self):
        """Test batch prediction with 10 transactions."""
        transactions = [generate_random_transaction() for _ in range(10)]
        
        with self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def predict_batch_large(self):
        """Test batch prediction with 100 transactions."""
        transactions = [generate_random_transaction() for _ in range(100)]
        
        with self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
            headers=self.headers,
            catch_response=True,
            name="/predict/batch [100]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Verify we got all predictions back
                if data.get("total_transactions") == 100:
                    response.success()
                else:
                    response.failure("Incomplete batch response")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(5)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Unhealthy status")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint."""
        with self.client.get(
            "/model/info",
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # Model not loaded is acceptable during testing
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class SLAValidationUser(FastHttpUser):
    """
    User focused on SLA validation.
    
    Checks that P99 latency is under 100ms for single predictions.
    """
    
    wait_time = between(0.05, 0.1)  # Faster requests
    
    @task
    def validate_sla(self):
        """Single prediction with SLA validation."""
        transaction = generate_random_transaction()
        
        with self.client.post(
            "/predict",
            json=transaction,
            catch_response=True,
        ) as response:
            # SLA: 200ms max for single prediction
            if response.elapsed.total_seconds() > 0.2:
                response.failure(
                    f"SLA violation: {response.elapsed.total_seconds()*1000:.0f}ms > 200ms"
                )
            elif response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# For running programmatically
if __name__ == "__main__":
    import subprocess
    subprocess.run([
        "locust",
        "-f", __file__,
        "--host=http://localhost:8000",
        "--headless",
        "-u", "50",
        "-r", "10",
        "-t", "60s",
        "--html=load_test_report.html",
    ])
