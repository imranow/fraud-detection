"""FastAPI application for fraud detection service."""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from fraud_detection import __version__
from fraud_detection.api.schemas import (
    BatchPredictionOutput,
    BatchTransactionInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionOutput,
    TransactionInput,
)
from fraud_detection.api.service import get_model_service, init_model_service
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of predictions made",
    ["result"],
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
FRAUD_DETECTED = Counter(
    "fraud_detected_total",
    "Total number of fraud cases detected",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting fraud detection API...")

    try:
        # Try to load model from environment or default path
        model_path = Path("models")
        model_files = list(model_path.glob("random_forest_*.joblib"))

        if model_files:
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            init_model_service(model_path=latest_model)
            logger.info(f"Model loaded: {latest_model}")
        else:
            logger.warning(
                "No model found. API will return errors until model is loaded."
            )
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

    yield

    # Shutdown
    logger.info("Shutting down fraud detection API...")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="""
    Production-grade API for real-time credit card fraud detection.

    ## Features
    - Single and batch transaction predictions
    - Risk level classification (LOW, MEDIUM, HIGH, CRITICAL)
    - Prometheus metrics for monitoring
    - Health checks for orchestration

    ## Model
    The API uses a Random Forest classifier trained on 284,807 transactions
    with 48 engineered features achieving:
    - **87.8% Recall** (catches most fraud)
    - **65.6% Precision** (low false positives)
    - **0.968 ROC-AUC**
    """,
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "Fraud Detection API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and orchestration."""
    service = get_model_service()

    return HealthResponse(
        status="healthy" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_name=service.model_name if service.is_loaded else "none",
        version=__version__,
        timestamp=datetime.now(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get information about the loaded model."""
    service = get_model_service()

    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=service.model_name,
        n_features=len(service.feature_names),
        feature_names=service.feature_names,
        threshold=service.threshold,
        training_metrics=service.training_metrics,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_transaction(
    transaction: TransactionInput,
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Custom threshold"
    ),
) -> PredictionOutput:
    """
    Predict fraud for a single transaction.

    Returns the fraud classification, probability, and risk level.
    """
    service = get_model_service()

    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Use custom threshold if provided
    if threshold is not None:
        original_threshold = service.threshold
        service.threshold = threshold

    start_time = time.time()

    try:
        is_fraud, probability, risk_level = service.predict_single(
            transaction.model_dump()
        )

        # Record metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS_TOTAL.labels(result="fraud" if is_fraud else "legit").inc()

        if is_fraud:
            FRAUD_DETECTED.inc()

        return PredictionOutput(
            is_fraud=is_fraud,
            fraud_probability=probability,
            risk_level=risk_level,
            threshold_used=service.threshold,
        )

    finally:
        # Restore original threshold
        if threshold is not None:
            service.threshold = original_threshold


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(
    batch: BatchTransactionInput,
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
) -> BatchPredictionOutput:
    """
    Predict fraud for multiple transactions in batch.

    Maximum 1000 transactions per request.
    """
    service = get_model_service()

    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if threshold is not None:
        original_threshold = service.threshold
        service.threshold = threshold

    start_time = time.time()

    try:
        transactions = [t.model_dump() for t in batch.transactions]
        results = service.predict(transactions)

        predictions = [
            PredictionOutput(
                is_fraud=is_fraud,
                fraud_probability=prob,
                risk_level=risk,
                threshold_used=service.threshold,
            )
            for is_fraud, prob, risk in results
        ]

        fraud_count = sum(1 for p in predictions if p.is_fraud)
        latency = (time.time() - start_time) * 1000  # Convert to ms

        # Record metrics
        PREDICTION_LATENCY.observe(latency / 1000)
        for p in predictions:
            PREDICTIONS_TOTAL.labels(result="fraud" if p.is_fraud else "legit").inc()
        FRAUD_DETECTED.inc(fraud_count)

        return BatchPredictionOutput(
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_detected=fraud_count,
            processing_time_ms=latency,
        )

    finally:
        if threshold is not None:
            service.threshold = original_threshold


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fraud_detection.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
