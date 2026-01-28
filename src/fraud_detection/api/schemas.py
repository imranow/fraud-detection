"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Input schema for a single transaction prediction."""

    # Time field (seconds from first transaction)
    Time: float = Field(..., description="Seconds since first transaction in dataset")

    # V1-V28 PCA components
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")

    # Amount
    Amount: float = Field(..., ge=0, description="Transaction amount")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807134,
                "V2": -0.072781173,
                "V3": 2.536346738,
                "V4": 1.378155224,
                "V5": -0.338321107,
                "V6": 0.462387778,
                "V7": 0.239598554,
                "V8": 0.098697901,
                "V9": 0.363787079,
                "V10": 0.090794172,
                "V11": -0.551599533,
                "V12": -0.617800856,
                "V13": -0.991389847,
                "V14": -0.311169354,
                "V15": 1.468176972,
                "V16": -0.470400525,
                "V17": 0.207971242,
                "V18": 0.02579058,
                "V19": 0.403992960,
                "V20": 0.251412098,
                "V21": -0.018306778,
                "V22": 0.277837576,
                "V23": -0.110473910,
                "V24": 0.066928075,
                "V25": 0.128539358,
                "V26": -0.189114844,
                "V27": 0.133558377,
                "V28": -0.021053053,
                "Amount": 149.62,
            }
        }


class BatchTransactionInput(BaseModel):
    """Input schema for batch predictions."""

    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to predict",
    )


class PredictionOutput(BaseModel):
    """Output schema for a single prediction."""

    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of fraud (0-1)",
    )
    risk_level: str = Field(
        ...,
        description="Risk level: LOW, MEDIUM, HIGH, CRITICAL",
    )
    threshold_used: float = Field(..., description="Classification threshold used")

    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": True,
                "fraud_probability": 0.847,
                "risk_level": "CRITICAL",
                "threshold_used": 0.28,
            }
        }


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""

    predictions: List[PredictionOutput]
    total_transactions: int
    fraud_detected: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Name of loaded model")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    n_features: int
    feature_names: List[str]
    threshold: float
    training_metrics: Dict[str, Any]
