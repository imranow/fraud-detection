# Fraud Detection System

A production-grade machine learning system for real-time credit card fraud detection.

## 🚀 Features

- **Real-time prediction** via REST API (FastAPI)
- **High accuracy** - 87.8% recall, 0.968 ROC-AUC
- **48 engineered features** from transaction data
- **MLOps ready** with MLflow experiment tracking
- **Production monitoring** with Prometheus metrics
- **Docker containerized** for easy deployment
- **Comprehensive testing** with pytest

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.968 |
| PR-AUC | 0.848 |
| Recall | 87.8% |
| Precision | 65.6% |
| F2 Score | 0.822 |

## 📁 Project Structure

```
fraud-detection/
├── src/fraud_detection/
│   ├── api/              # FastAPI endpoints
│   ├── data/             # Data loading & preprocessing
│   ├── features/         # Feature engineering (48 features)
│   ├── models/           # Model trainers
│   ├── evaluation/       # Metrics & evaluation
│   └── utils/            # Utilities
├── models/               # Saved model artifacts
├── configs/              # Hydra configuration
├── scripts/              # Training & deployment scripts
├── docker/               # Docker configurations
├── Dockerfile            # Container definition
└── docker-compose.yml    # Multi-container setup
```

## 🛠️ Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# Or use the script
chmod +x scripts/docker-run.sh
./scripts/docker-run.sh
```

API available at: http://localhost:8000

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API
PYTHONPATH=src uvicorn fraud_detection.api.main:app --reload
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.359, "V2": -0.072, "V3": 2.536,
    ...
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
```

## 📈 Monitoring

With Docker Compose monitoring profile:

```bash
docker-compose --profile monitoring up -d
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

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
