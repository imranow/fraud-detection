"""Base model trainer and utilities for fraud detection.

This module provides abstract base classes and utilities for
training fraud detection models with proper handling of class imbalance.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for model trainers.
    
    Provides common functionality for training, evaluation, and persistence.
    """

    def __init__(
        self,
        model_name: str,
        random_state: int = 42,
        model_dir: Optional[Path] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name identifier for the model
            random_state: Random seed for reproducibility
            model_dir: Directory to save model artifacts
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model_dir = model_dir or Path("models")
        
        self._model: Optional[BaseEstimator] = None
        self._is_fitted: bool = False
        self._training_metrics: Dict[str, Any] = {}
        self._feature_names: List[str] = []

    @abstractmethod
    def _create_model(self, **kwargs) -> BaseEstimator:
        """Create the underlying model instance."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for the model."""
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> "BaseTrainer":
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Optional (X_val, y_val) for validation
            **kwargs: Additional arguments for the model
            
        Returns:
            self
        """
        logger.info(f"Training {self.model_name} on {len(X):,} samples")
        
        self._feature_names = list(X.columns)
        self._model = self._create_model(**kwargs)
        
        # Prepare fit arguments
        fit_kwargs = self._prepare_fit_kwargs(X, y, eval_set, **kwargs)
        
        # Train the model
        start_time = datetime.now()
        self._model.fit(X, y, **fit_kwargs)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self._is_fitted = True
        self._training_metrics["training_time_seconds"] = training_time
        self._training_metrics["n_samples"] = len(X)
        self._training_metrics["n_features"] = len(self._feature_names)
        
        logger.info(f"Training complete in {training_time:.2f}s")
        
        return self

    def _prepare_fit_kwargs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare arguments for model.fit(). Override in subclasses."""
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores if available."""
        if not self._is_fitted:
            return None
        
        if hasattr(self._model, "feature_importances_"):
            return pd.Series(
                self._model.feature_importances_,
                index=self._feature_names,
            ).sort_values(ascending=False)
        return None

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save the trained model.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved model
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.joblib"
        
        filepath = self.model_dir / filename
        
        model_artifact = {
            "model": self._model,
            "model_name": self.model_name,
            "feature_names": self._feature_names,
            "training_metrics": self._training_metrics,
            "random_state": self.random_state,
        }
        
        joblib.dump(model_artifact, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseTrainer":
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded trainer instance
        """
        filepath = Path(filepath)
        artifact = joblib.load(filepath)
        
        # Create a new instance
        trainer = cls.__new__(cls)
        trainer.model_name = artifact["model_name"]
        trainer._model = artifact["model"]
        trainer._feature_names = artifact["feature_names"]
        trainer._training_metrics = artifact["training_metrics"]
        trainer.random_state = artifact["random_state"]
        trainer.model_dir = filepath.parent
        trainer._is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return trainer

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation prediction.
        
        Args:
            X: Features
            y: Labels
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with predictions and probabilities
        """
        logger.info(f"Running {n_folds}-fold cross-validation")
        
        model = self._create_model(**kwargs)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Get probability predictions
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)
        
        return {
            "y_pred": y_pred,
            "y_proba": y_proba[:, 1],
            "y_true": y.values,
        }

    @property
    def model(self) -> Optional[BaseEstimator]:
        """Get the underlying model."""
        return self._model

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self._model is not None:
            return self._model.get_params()
        return self.get_default_params()
