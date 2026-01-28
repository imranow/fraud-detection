"""Feature engineering pipeline for fraud detection.

This module provides a complete feature engineering pipeline that
combines all transformers into a single sklearn Pipeline.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_detection.features.transformers import (
    AmountFeatures,
    AnomalyScoreFeatures,
    InteractionFeatures,
    TimeFeatures,
    VelocityFeatures,
)
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Complete feature engineering pipeline for fraud detection.

    This class orchestrates all feature transformers and provides
    a unified interface for feature engineering.

    Example:
        >>> engineer = FeatureEngineer()
        >>> X_train_fe = engineer.fit_transform(X_train, y_train)
        >>> X_test_fe = engineer.transform(X_test)
    """

    # Original V columns (from PCA)
    V_COLUMNS = [f"V{i}" for i in range(1, 29)]

    # Base columns
    BASE_COLUMNS = V_COLUMNS + ["Amount", "Time"]

    def __init__(
        self,
        include_amount_features: bool = True,
        include_time_features: bool = True,
        include_velocity_features: bool = True,
        include_interaction_features: bool = True,
        include_anomaly_features: bool = True,
        scale_new_features: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize the feature engineer.

        Args:
            include_amount_features: Create amount-based features
            include_time_features: Create time-based features
            include_velocity_features: Create velocity/statistical features
            include_interaction_features: Create interaction features
            include_anomaly_features: Create anomaly score features
            scale_new_features: Standardize newly created features
            random_state: Random seed for reproducibility
        """
        self.include_amount_features = include_amount_features
        self.include_time_features = include_time_features
        self.include_velocity_features = include_velocity_features
        self.include_interaction_features = include_interaction_features
        self.include_anomaly_features = include_anomaly_features
        self.scale_new_features = scale_new_features
        self.random_state = random_state

        self._transformers: List[Any] = []
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._is_fitted: bool = False

    def _create_transformers(self) -> List:
        """Create the list of feature transformers."""
        transformers = []

        if self.include_amount_features:
            transformers.append(("amount", AmountFeatures()))

        if self.include_time_features:
            transformers.append(("time", TimeFeatures()))

        if self.include_velocity_features:
            transformers.append(("velocity", VelocityFeatures()))

        if self.include_interaction_features:
            transformers.append(("interaction", InteractionFeatures()))

        if self.include_anomaly_features:
            transformers.append(("anomaly", AnomalyScoreFeatures()))

        return transformers

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Fit the feature engineering pipeline.

        Args:
            X: Input features DataFrame
            y: Target labels (optional, used for anomaly features)

        Returns:
            self
        """
        logger.info(f"Fitting feature engineer on {len(X):,} samples")

        self._transformers = self._create_transformers()

        # Fit each transformer
        for name, transformer in self._transformers:
            logger.info(f"Fitting {name} transformer")
            if hasattr(transformer, "fit"):
                transformer.fit(X, y)

        # Collect feature names
        self._collect_feature_names(X)

        # Fit scaler on a sample transform
        if self.scale_new_features:
            X_transformed = self._apply_transformers(X)
            new_cols = [c for c in X_transformed.columns if c not in self.BASE_COLUMNS]
            if new_cols:
                self._scaler = StandardScaler()
                self._scaler.fit(X_transformed[new_cols])

        self._is_fitted = True
        logger.info(
            f"Feature engineer fitted. Output features: {len(self._feature_names)}"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted transformers.

        Args:
            X: Input features DataFrame

        Returns:
            Transformed DataFrame with engineered features
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        X_transformed = self._apply_transformers(X)

        # Scale new features
        if self.scale_new_features and self._scaler is not None:
            new_cols = [c for c in X_transformed.columns if c not in self.BASE_COLUMNS]
            if new_cols:
                X_transformed[new_cols] = self._scaler.transform(
                    X_transformed[new_cols]
                )

        return X_transformed

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Input features DataFrame
            y: Target labels (optional)

        Returns:
            Transformed DataFrame with engineered features
        """
        return self.fit(X, y).transform(X)

    def _apply_transformers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformers to the data."""
        X_result = X.copy()

        for name, transformer in self._transformers:
            X_result = transformer.transform(X_result)

        return X_result

    def _collect_feature_names(self, X: pd.DataFrame) -> None:
        """Collect all feature names after transformation."""
        X_sample = self._apply_transformers(X.head(100))
        self._feature_names = list(X_sample.columns)

    def get_feature_names(self) -> List[str]:
        """Get the names of all output features."""
        return self._feature_names

    def get_new_feature_names(self) -> List[str]:
        """Get names of newly created features (not in original data)."""
        return [f for f in self._feature_names if f not in self.BASE_COLUMNS]

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the feature engineer."""
        return {
            "include_amount_features": self.include_amount_features,
            "include_time_features": self.include_time_features,
            "include_velocity_features": self.include_velocity_features,
            "include_interaction_features": self.include_interaction_features,
            "include_anomaly_features": self.include_anomaly_features,
            "scale_new_features": self.scale_new_features,
            "random_state": self.random_state,
            "n_output_features": len(self._feature_names),
            "n_new_features": len(self.get_new_feature_names()),
        }


def create_feature_pipeline(
    include_amount: bool = True,
    include_time: bool = True,
    include_velocity: bool = True,
    include_interaction: bool = True,
    include_anomaly: bool = True,
) -> FeatureEngineer:
    """
    Create a feature engineering pipeline with specified components.

    Args:
        include_amount: Include amount-based features
        include_time: Include time-based features
        include_velocity: Include velocity/statistical features
        include_interaction: Include interaction features
        include_anomaly: Include anomaly score features

    Returns:
        Configured FeatureEngineer instance
    """
    return FeatureEngineer(
        include_amount_features=include_amount,
        include_time_features=include_time,
        include_velocity_features=include_velocity,
        include_interaction_features=include_interaction,
        include_anomaly_features=include_anomaly,
    )
