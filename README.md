# Fraud Detection System

A production-grade machine learning system for real-time credit card fraud detection.

## 🚀 Features

- **Real-time prediction** via REST API (FastAPI)
- **High accuracy** - 87.8% recall, 0.968 ROC-AUC
- **48 engineered features** from transaction data
- **API authentication** with secure key management
- **Rate limiting** with Redis-backed token bucket
- **A/B testing** infrastructure for model comparison
- **MLOps ready** with MLflow experiment tracking
- **Production monitoring** with Prometheus & Grafana
- **Log aggregation** with Loki
- **Docker containerized** for easy deployment
- **Kubernetes ready** with full manifests
- **Comprehensive testing** with pytest & Locust load tests

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.968 |
| PR-AUC | 0.848 |
| Recall | 87.8% |
| Precision | 65.6% |
| F2 Score | 0.822 |

## 📥 Getting the Data

The dataset is the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (144MB). It's not included in the repository due to size limits.

### Option 1: Download Script (Recommended)

```bash
python scripts/download_data.py
```

This will:
- Try Kaggle API first (if configured)
- Fall back to OpenML mirror
- Verify the downloaded file

### Option 2: Manual Download

1. Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in `data/raw/`

### Option 3: DVC (For Teams)

If DVC remote storage is configured:
```bash
dvc pull
```

## 📁 Project Structure

```
fraud-detection/
├── src/fraud_detection/
│   ├── api/              # FastAPI endpoints + auth + rate limiting
│   ├── data/             # Data loading & preprocessing
│   ├── features/         # Feature engineering + feature store
│   ├── models/           # Model trainers
│   ├── evaluation/       # Metrics & evaluation
│   └── utils/            # Utilities + MLflow config
├── models/               # Saved model artifacts
├── configs/              # Hydra configuration
├── scripts/              # Training & deployment scripts
├── k8s/                  # Kubernetes manifests
├── prometheus/           # Prometheus config & alerting rules
├── grafana/              # Grafana dashboards & provisioning
├── loki/                 # Loki & Promtail configuration
├── tests/                # Unit tests + load tests
├── notebooks/            # EDA & analysis notebooks
├── Dockerfile            # Container definition
└── docker-compose.yml    # Multi-container setup
```

## 🛠️ Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# Or with full monitoring stack
docker-compose --profile monitoring up -d
```

API available at: http://localhost:8000

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# Run API
PYTHONPATH=src uvicorn fraud_detection.api.main:app --reload
```

## 🔐 API Authentication

Protected endpoints require an API key:

```bash
# Generate a key
source .venv/bin/activate
PYTHONPATH=src python -c "from fraud_detection.api.auth import generate_api_key; print(generate_api_key())"

# Use in requests
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"Time": 0, "V1": -1.36, ...}'
```

Set the `API_KEY` environment variable to enable authentication.

## 📡 API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/predict` | POST | Yes* | Single prediction |
| `/predict/batch` | POST | Yes* | Batch predictions |
| `/model/info` | GET | Yes* | Model metadata |
| `/metrics` | GET | No | Prometheus metrics |
| `/docs` | GET | No | Swagger UI |

*Auth required only when `API_KEY` is configured

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
    "V5": -0.338, "V6": 0.462, "V7": 0.240, "V8": 0.099,
    "V9": 0.364, "V10": 0.091, "V11": -0.552, "V12": -0.618,
    "V13": -0.991, "V14": -0.311, "V15": 1.468, "V16": -0.470,
    "V17": 0.208, "V18": 0.026, "V19": 0.404, "V20": 0.251,
    "V21": -0.018, "V22": 0.278, "V23": -0.110, "V24": 0.067,
    "V25": 0.129, "V26": -0.189, "V27": 0.134, "V28": -0.021,
    "Amount": 149.62
  }'
```

### Example Response

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0049,
  "risk_level": "LOW",
  "threshold_used": 0.28
}
```

## 🧪 Training

```bash
# Train all models with feature engineering
python scripts/train.py --model all --feature-engineering --select-features

# Train specific model
python scripts/train.py --model random_forest

# Automated retraining with drift detection
python scripts/retrain.py
```

## 📈 Monitoring

With Docker Compose monitoring profile:

```bash
docker-compose --profile monitoring up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Loki** | http://localhost:3100 | - |
| **Alertmanager** | http://localhost:9093 | - |

## ☸️ Kubernetes Deployment

```bash
# Deploy everything
./scripts/k8s-deploy.sh deploy

# Or using kubectl
kubectl apply -k k8s/

# Port forward for local access
./scripts/k8s-deploy.sh forward
```

## 🔥 Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 🐳 Docker Commands

```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -p 8000:8000 fraud-detection:latest

# View logs
docker logs fraud-detection-api

# Stop
docker-compose down
```

## 📝 License

MIT License
