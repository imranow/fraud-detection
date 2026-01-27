"""Model service for loading and inference."""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from fraud_detection.features import FeatureEngineer, FeatureSelector
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Service for loading models and making predictions.
    
    Handles model loading, feature engineering, and inference
    for the fraud detection API.
    """

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "LOW": 0.2,
        "MEDIUM": 0.5,
        "HIGH": 0.7,
        "CRITICAL": 0.85,
    }

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: float = 0.28,  # Optimal from training
    ):
        """
        Initialize the model service.
        
        Args:
            model_path: Path to the model file
            threshold: Classification threshold
        """
        self.model_path = model_path
        self.threshold = threshold
        
        self._model = None
        self._model_name: str = ""
        self._feature_names: List[str] = []
        self._training_metrics: dict = {}
        self._feature_engineer: Optional[FeatureEngineer] = None
        self._feature_selector: Optional[FeatureSelector] = None
        self._is_loaded: bool = False

    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load the model from disk.
        
        Args:
            model_path: Optional override path
        """
        path = model_path or self.model_path
        
        if path is None:
            # Try to find the best model in models directory
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("random_forest_*.joblib"))
                if model_files:
                    path = sorted(model_files)[-1]  # Latest model
                    logger.info(f"Auto-selected model: {path}")
        
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        logger.info(f"Loading model from {path}")
        
        artifact = joblib.load(path)
        
        self._model = artifact["model"]
        self._model_name = artifact.get("model_name", "unknown")
        self._feature_names = artifact.get("feature_names", [])
        self._training_metrics = artifact.get("training_metrics", {})
        
        # Initialize feature engineer for consistent preprocessing
        self._feature_engineer = FeatureEngineer()
        
        self._is_loaded = True
        logger.info(f"Model loaded: {self._model_name} with {len(self._feature_names)} features")

    def predict(
        self,
        transactions: List[dict],
    ) -> List[Tuple[bool, float, str]]:
        """
        Make predictions on transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of (is_fraud, probability, risk_level) tuples
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Apply feature engineering
        if self._feature_engineer is not None:
            try:
                df_fe = self._feature_engineer.fit_transform(df)
            except Exception as e:
                logger.warning(f"Feature engineering failed, using raw features: {e}")
                df_fe = df
        else:
            df_fe = df
        
        # Select only the features the model was trained on
        missing_features = set(self._feature_names) - set(df_fe.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features as zeros (not ideal but prevents errors)
            for feat in missing_features:
                df_fe[feat] = 0
        
        # Ensure correct column order
        X = df_fe[self._feature_names]
        
        # Get probabilities
        probabilities = self._model.predict_proba(X)[:, 1]
        
        # Create predictions
        results = []
        for prob in probabilities:
            is_fraud = prob >= self.threshold
            risk_level = self._get_risk_level(prob)
            results.append((is_fraud, float(prob), risk_level))
        
        return results

    def predict_single(self, transaction: dict) -> Tuple[bool, float, str]:
        """
        Make prediction on a single transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Tuple of (is_fraud, probability, risk_level)
        """
        results = self.predict([transaction])
        return results[0]

    def _get_risk_level(self, probability: float) -> str:
        """Get risk level from probability."""
        if probability >= self.RISK_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif probability >= self.RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif probability >= self.RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names

    @property
    def training_metrics(self) -> dict:
        """Get training metrics."""
        return self._training_metrics


# Global model service instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def init_model_service(model_path: Optional[Path] = None, threshold: float = 0.28) -> ModelService:
    """Initialize the global model service."""
    global _model_service
    _model_service = ModelService(model_path=model_path, threshold=threshold)
    _model_service.load()
    return _model_service
