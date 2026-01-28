"""API module for fraud detection service."""

from fraud_detection.api.main import app
from fraud_detection.api.schemas import (
    BatchPredictionOutput,
    BatchTransactionInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionOutput,
    TransactionInput,
)
from fraud_detection.api.service import (
    ModelService,
    get_model_service,
    init_model_service,
)

__all__ = [
    "app",
    "ModelService",
    "get_model_service",
    "init_model_service",
    "TransactionInput",
    "BatchTransactionInput",
    "PredictionOutput",
    "BatchPredictionOutput",
    "HealthResponse",
    "ModelInfoResponse",
]
