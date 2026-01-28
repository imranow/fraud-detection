"""Feature engineering transformers for fraud detection.

This module provides sklearn-compatible transformers for creating
features from transaction data.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmountFeatures(BaseEstimator, TransformerMixin):
    """Create features based on transaction amount.

    Features created:
    - Amount_log: Log-transformed amount
    - Amount_log1p: Log(1+amount) transformation
    - Amount_sqrt: Square root of amount
    - Amount_bin: Binned amount categories
    - Amount_is_round: Whether amount is a round number
    - Amount_cents: Cents portion of the amount
    """

    def __init__(self, create_bins: bool = True, n_bins: int = 10):
        self.create_bins = create_bins
        self.n_bins = n_bins
        self._bin_edges: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AmountFeatures":
        """Fit the transformer, learning bin edges if needed."""
        if self.create_bins and "Amount" in X.columns:
            # Use quantile-based binning
            self._bin_edges = np.percentile(
                X["Amount"].values, np.linspace(0, 100, self.n_bins + 1)
            )
            # Ensure unique edges
            self._bin_edges = np.unique(self._bin_edges)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding amount-based features."""
        X = X.copy()

        if "Amount" not in X.columns:
            return X

        amount = X["Amount"]

        # Log transformations (handle zeros)
        X["Amount_log"] = np.log(amount.clip(lower=1e-10))
        X["Amount_log1p"] = np.log1p(amount)
        X["Amount_sqrt"] = np.sqrt(amount)

        # Round number detection
        X["Amount_is_round"] = (amount % 1 == 0).astype(int)
        X["Amount_is_round_10"] = (amount % 10 == 0).astype(int)
        X["Amount_is_round_100"] = (amount % 100 == 0).astype(int)

        # Cents extraction
        X["Amount_cents"] = (amount * 100) % 100

        # Binned amount
        if self.create_bins and self._bin_edges is not None:
            X["Amount_bin"] = np.digitize(amount, self._bin_edges[1:-1])

        return X

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        new_features = [
            "Amount_log",
            "Amount_log1p",
            "Amount_sqrt",
            "Amount_is_round",
            "Amount_is_round_10",
            "Amount_is_round_100",
            "Amount_cents",
        ]
        if self.create_bins:
            new_features.append("Amount_bin")
        return new_features


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Create features based on transaction time.

    The Time column represents seconds elapsed from the first transaction.

    Features created:
    - Hour_of_day: Hour within a 24-hour cycle
    - Time_sin/cos: Cyclical encoding of time
    - Is_night: Whether transaction occurred during night hours
    - Is_weekend: Whether transaction is on weekend (approximate)
    """

    def __init__(self) -> None:
        self._time_min: float = 0
        self._time_max: float = 0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TimeFeatures":
        """Fit the transformer."""
        if "Time" in X.columns:
            self._time_min = X["Time"].min()
            self._time_max = X["Time"].max()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding time-based features."""
        X = X.copy()

        if "Time" not in X.columns:
            return X

        time = X["Time"]
        seconds_per_day = 86400  # 24 * 60 * 60
        seconds_per_hour = 3600

        # Hour of day (assuming time starts at midnight)
        X["Hour_of_day"] = (time % seconds_per_day) / seconds_per_hour

        # Cyclical encoding for smooth transitions
        X["Time_sin"] = np.sin(2 * np.pi * X["Hour_of_day"] / 24)
        X["Time_cos"] = np.cos(2 * np.pi * X["Hour_of_day"] / 24)

        # Night hours (11 PM - 6 AM)
        hour = X["Hour_of_day"]
        X["Is_night"] = ((hour >= 23) | (hour <= 6)).astype(int)

        # Business hours (9 AM - 5 PM)
        X["Is_business_hours"] = ((hour >= 9) & (hour <= 17)).astype(int)

        # Day within the dataset (elapsed days)
        X["Day_elapsed"] = time // seconds_per_day

        # Approximate weekend detection (every 5th and 6th day, or day 0 as Sunday)
        day_of_week = (time // seconds_per_day) % 7
        X["Is_weekend"] = ((day_of_week >= 5) | (day_of_week == 0)).astype(int)

        return X

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        return [
            "Hour_of_day",
            "Time_sin",
            "Time_cos",
            "Is_night",
            "Is_business_hours",
            "Day_elapsed",
            "Is_weekend",
        ]


class VelocityFeatures(BaseEstimator, TransformerMixin):
    """Create velocity/frequency features based on PCA components.

    Since we don't have customer IDs, we create statistical features
    from the V1-V28 components that may capture unusual patterns.

    Features created:
    - V_magnitude: L2 norm of V1-V28
    - V_mean: Mean of V1-V28
    - V_std: Standard deviation of V1-V28
    - V_max/V_min: Max and min of V components
    - V_skew: Skewness across V components
    - V_kurtosis: Kurtosis across V components
    - V_n_outliers: Number of extreme V values
    """

    def __init__(self, outlier_threshold: float = 3.0):
        self.outlier_threshold = outlier_threshold
        self._v_means: Optional[np.ndarray] = None
        self._v_stds: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "VelocityFeatures":
        """Fit the transformer, learning V component statistics."""
        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in X.columns]
        if v_cols:
            v_data = X[v_cols].values
            self._v_means = v_data.mean(axis=0)
            self._v_stds = v_data.std(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding velocity features."""
        X = X.copy()

        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in X.columns]
        if not v_cols:
            return X

        v_data = X[v_cols].values

        # Magnitude (L2 norm)
        X["V_magnitude"] = np.linalg.norm(v_data, axis=1)

        # Basic statistics across V components
        X["V_mean"] = v_data.mean(axis=1)
        X["V_std"] = v_data.std(axis=1)
        X["V_max"] = v_data.max(axis=1)
        X["V_min"] = v_data.min(axis=1)
        X["V_range"] = X["V_max"] - X["V_min"]

        # Higher-order statistics
        X["V_skew"] = pd.DataFrame(v_data).apply(lambda row: row.skew(), axis=1).values
        X["V_kurtosis"] = (
            pd.DataFrame(v_data).apply(lambda row: row.kurtosis(), axis=1).values
        )

        # Outlier detection (how many V components are extreme)
        if self._v_stds is not None and self._v_means is not None:
            z_scores = np.abs((v_data - self._v_means) / (self._v_stds + 1e-10))
            X["V_n_outliers"] = (z_scores > self.outlier_threshold).sum(axis=1)
            X["V_max_zscore"] = z_scores.max(axis=1)

        # Key individual V components that are often important for fraud
        # (V14, V17, V12, V10 typically have high importance)
        for v in ["V14", "V17", "V12", "V10", "V4", "V11"]:
            if v in X.columns:
                X[f"{v}_abs"] = np.abs(X[v])

        return X

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        return [
            "V_magnitude",
            "V_mean",
            "V_std",
            "V_max",
            "V_min",
            "V_range",
            "V_skew",
            "V_kurtosis",
            "V_n_outliers",
            "V_max_zscore",
            "V14_abs",
            "V17_abs",
            "V12_abs",
            "V10_abs",
            "V4_abs",
            "V11_abs",
        ]


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Create interaction features between key variables.

    Features created:
    - Amount * V component interactions
    - V component ratios
    - Polynomial features for key variables
    """

    def __init__(self, top_v_components: Optional[List[str]] = None):
        self.top_v_components = top_v_components or ["V14", "V17", "V12", "V10"]

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "InteractionFeatures":
        """Fit the transformer."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding interaction features."""
        X = X.copy()

        # Amount interactions with key V components
        if "Amount" in X.columns:
            amount_log = np.log1p(X["Amount"])
            for v in self.top_v_components:
                if v in X.columns:
                    X[f"Amount_x_{v}"] = amount_log * X[v]

        # Ratios between key V components
        for i, v1 in enumerate(self.top_v_components[:-1]):
            for v2 in self.top_v_components[i + 1 :]:
                if v1 in X.columns and v2 in X.columns:
                    # Add small epsilon to avoid division by zero
                    X[f"{v1}_div_{v2}"] = X[v1] / (X[v2].abs() + 1e-10)

        # Squared terms for key components
        for v in self.top_v_components[:2]:
            if v in X.columns:
                X[f"{v}_squared"] = X[v] ** 2

        return X

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        features = []
        for v in self.top_v_components:
            features.append(f"Amount_x_{v}")
        for i, v1 in enumerate(self.top_v_components[:-1]):
            for v2 in self.top_v_components[i + 1 :]:
                features.append(f"{v1}_div_{v2}")
        for v in self.top_v_components[:2]:
            features.append(f"{v}_squared")
        return features


class AnomalyScoreFeatures(BaseEstimator, TransformerMixin):
    """Create isolation-based anomaly score features.

    Uses distance from cluster centroids and statistical measures
    to create anomaly indicators.
    """

    def __init__(self) -> None:
        self._centroid_normal: Optional[np.ndarray] = None
        self._centroid_fraud: Optional[np.ndarray] = None
        self._overall_mean: Optional[np.ndarray] = None
        self._overall_std: Optional[np.ndarray] = None

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AnomalyScoreFeatures":
        """Fit the transformer, computing centroids if labels provided."""
        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in X.columns]
        if not v_cols:
            return self

        v_data = X[v_cols].values
        self._overall_mean = v_data.mean(axis=0)
        self._overall_std = v_data.std(axis=0)

        if y is not None:
            y_arr = np.array(y)
            self._centroid_normal = v_data[y_arr == 0].mean(axis=0)
            self._centroid_fraud = v_data[y_arr == 1].mean(axis=0)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding anomaly score features."""
        X = X.copy()

        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in X.columns]
        if not v_cols or self._overall_mean is None:
            return X

        v_data = X[v_cols].values

        # Distance from overall mean (Mahalanobis-like)
        assert self._overall_std is not None
        z_scores = (v_data - self._overall_mean) / (self._overall_std + 1e-10)
        X["Anomaly_zscore_sum"] = np.abs(z_scores).sum(axis=1)
        X["Anomaly_zscore_max"] = np.abs(z_scores).max(axis=1)

        # Distance from centroids (if available)
        if self._centroid_normal is not None:
            X["Dist_to_normal"] = np.linalg.norm(v_data - self._centroid_normal, axis=1)
        if self._centroid_fraud is not None:
            X["Dist_to_fraud"] = np.linalg.norm(v_data - self._centroid_fraud, axis=1)
            # Ratio of distances
            if "Dist_to_normal" in X.columns:
                X["Dist_ratio"] = X["Dist_to_fraud"] / (X["Dist_to_normal"] + 1e-10)

        return X

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        features = ["Anomaly_zscore_sum", "Anomaly_zscore_max"]
        if self._centroid_normal is not None:
            features.append("Dist_to_normal")
        if self._centroid_fraud is not None:
            features.extend(["Dist_to_fraud", "Dist_ratio"])
        return features
