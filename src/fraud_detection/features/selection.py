"""Feature selection utilities for fraud detection.

This module provides feature selection methods optimized for
imbalanced classification tasks.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_classif,
)

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection for fraud detection.
    
    Supports multiple selection methods:
    - importance: Based on tree model feature importance
    - mutual_info: Based on mutual information with target
    - correlation: Remove highly correlated features
    - combined: Use multiple methods together
    """

    def __init__(
        self,
        method: str = "importance",
        n_features: Optional[int] = None,
        importance_threshold: float = 0.001,
        correlation_threshold: float = 0.95,
        random_state: int = 42,
    ):
        """
        Initialize the feature selector.
        
        Args:
            method: Selection method ('importance', 'mutual_info', 'correlation', 'combined')
            n_features: Number of features to select (if None, use threshold)
            importance_threshold: Minimum importance score to keep feature
            correlation_threshold: Maximum correlation between features
            random_state: Random seed
        """
        self.method = method
        self.n_features = n_features
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        
        self._selected_features: List[str] = []
        self._feature_importances: Optional[pd.Series] = None
        self._dropped_correlated: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """
        Fit the feature selector.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            
        Returns:
            self
        """
        logger.info(f"Fitting feature selector with method: {self.method}")
        
        if self.method == "importance":
            self._fit_importance(X, y)
        elif self.method == "mutual_info":
            self._fit_mutual_info(X, y)
        elif self.method == "correlation":
            self._fit_correlation(X)
        elif self.method == "combined":
            self._fit_combined(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Selected {len(self._selected_features)} features")
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with selected features only
        """
        return X[self._selected_features].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _fit_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Select features based on Random Forest importance."""
        # Train a quick Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        self._feature_importances = importances
        
        # Select features
        if self.n_features:
            self._selected_features = importances.head(self.n_features).index.tolist()
        else:
            self._selected_features = importances[
                importances >= self.importance_threshold
            ].index.tolist()

    def _fit_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Select features based on mutual information."""
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        self._feature_importances = mi_series
        
        if self.n_features:
            self._selected_features = mi_series.head(self.n_features).index.tolist()
        else:
            # Use relative threshold
            threshold = mi_series.max() * 0.01
            self._selected_features = mi_series[mi_series >= threshold].index.tolist()

    def _fit_correlation(self, X: pd.DataFrame) -> None:
        """Remove highly correlated features."""
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > self.correlation_threshold].tolist()
            to_drop.update(correlated)
        
        self._dropped_correlated = list(to_drop)
        self._selected_features = [c for c in X.columns if c not in to_drop]

    def _fit_combined(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Use multiple methods together."""
        # First remove highly correlated
        self._fit_correlation(X)
        X_reduced = X[self._selected_features]
        
        # Then select by importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_reduced, y)
        
        importances = pd.Series(rf.feature_importances_, index=X_reduced.columns)
        importances = importances.sort_values(ascending=False)
        self._feature_importances = importances
        
        if self.n_features:
            self._selected_features = importances.head(self.n_features).index.tolist()
        else:
            self._selected_features = importances[
                importances >= self.importance_threshold
            ].index.tolist()

    def get_feature_importances(self) -> Optional[pd.Series]:
        """Get feature importance scores."""
        return self._feature_importances

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self._selected_features

    def get_dropped_features(self) -> List[str]:
        """Get list of dropped feature names."""
        if self._feature_importances is not None:
            all_features = set(self._feature_importances.index)
            return list(all_features - set(self._selected_features))
        return []


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    method: str = "importance",
    n_features: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], FeatureSelector]:
    """
    Convenience function to select features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        method: Selection method
        n_features: Number of features to keep
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selector)
    """
    selector = FeatureSelector(method=method, n_features=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    X_test_selected = None
    if X_test is not None:
        X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector
