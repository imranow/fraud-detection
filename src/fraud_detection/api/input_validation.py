"""Enhanced input validation with anomaly detection for fraud API."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureBounds:
    """Statistical bounds for a feature."""
    
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    
    # How many standard deviations before flagging as anomaly
    n_sigma: float = 3.0
    
    @property
    def lower_bound(self) -> float:
        return max(self.min_val, self.mean - self.n_sigma * self.std)
    
    @property
    def upper_bound(self) -> float:
        return min(self.max_val, self.mean + self.n_sigma * self.std)
    
    def is_within_bounds(self, value: float) -> bool:
        """Check if value is within expected bounds."""
        return self.lower_bound <= value <= self.upper_bound
    
    def get_deviation(self, value: float) -> float:
        """Get number of standard deviations from mean."""
        if self.std == 0:
            return 0.0
        return abs(value - self.mean) / self.std


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    anomalies: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0
    
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
    
    def add_anomaly(self, feature: str, deviation: float) -> None:
        self.anomalies[feature] = deviation


# Pre-computed statistics from training data (V1-V28 are PCA-transformed)
# These bounds are based on the credit card fraud dataset
DEFAULT_FEATURE_BOUNDS = {
    "Time": FeatureBounds("Time", mean=94813.0, std=47488.0, min_val=0.0, max_val=172800.0),
    "Amount": FeatureBounds("Amount", mean=88.35, std=250.12, min_val=0.0, max_val=25691.16),
    "V1": FeatureBounds("V1", mean=0.0, std=1.96, min_val=-56.41, max_val=2.45),
    "V2": FeatureBounds("V2", mean=0.0, std=1.65, min_val=-72.72, max_val=22.06),
    "V3": FeatureBounds("V3", mean=0.0, std=1.52, min_val=-48.33, max_val=9.38),
    "V4": FeatureBounds("V4", mean=0.0, std=1.42, min_val=-5.68, max_val=16.88),
    "V5": FeatureBounds("V5", mean=0.0, std=1.38, min_val=-113.74, max_val=34.80),
    "V6": FeatureBounds("V6", mean=0.0, std=1.33, min_val=-26.16, max_val=73.30),
    "V7": FeatureBounds("V7", mean=0.0, std=1.24, min_val=-43.56, max_val=120.59),
    "V8": FeatureBounds("V8", mean=0.0, std=1.19, min_val=-73.22, max_val=20.01),
    "V9": FeatureBounds("V9", mean=0.0, std=1.10, min_val=-13.43, max_val=15.59),
    "V10": FeatureBounds("V10", mean=0.0, std=1.09, min_val=-24.59, max_val=23.75),
    "V11": FeatureBounds("V11", mean=0.0, std=1.02, min_val=-4.80, max_val=12.02),
    "V12": FeatureBounds("V12", mean=0.0, std=1.00, min_val=-18.68, max_val=7.85),
    "V13": FeatureBounds("V13", mean=0.0, std=1.00, min_val=-5.79, max_val=7.13),
    "V14": FeatureBounds("V14", mean=0.0, std=0.96, min_val=-19.21, max_val=10.53),
    "V15": FeatureBounds("V15", mean=0.0, std=0.92, min_val=-4.50, max_val=8.88),
    "V16": FeatureBounds("V16", mean=0.0, std=0.88, min_val=-14.13, max_val=17.32),
    "V17": FeatureBounds("V17", mean=0.0, std=0.85, min_val=-25.16, max_val=9.25),
    "V18": FeatureBounds("V18", mean=0.0, std=0.84, min_val=-9.50, max_val=5.04),
    "V19": FeatureBounds("V19", mean=0.0, std=0.81, min_val=-7.21, max_val=5.59),
    "V20": FeatureBounds("V20", mean=0.0, std=0.77, min_val=-54.50, max_val=39.42),
    "V21": FeatureBounds("V21", mean=0.0, std=0.73, min_val=-34.83, max_val=27.20),
    "V22": FeatureBounds("V22", mean=0.0, std=0.73, min_val=-10.93, max_val=10.50),
    "V23": FeatureBounds("V23", mean=0.0, std=0.62, min_val=-44.81, max_val=22.53),
    "V24": FeatureBounds("V24", mean=0.0, std=0.61, min_val=-2.84, max_val=4.58),
    "V25": FeatureBounds("V25", mean=0.0, std=0.52, min_val=-10.30, max_val=7.52),
    "V26": FeatureBounds("V26", mean=0.0, std=0.48, min_val=-2.60, max_val=3.52),
    "V27": FeatureBounds("V27", mean=0.0, std=0.40, min_val=-22.57, max_val=31.61),
    "V28": FeatureBounds("V28", mean=0.0, std=0.33, min_val=-15.43, max_val=33.85),
}


class InputValidator:
    """Validate and detect anomalies in transaction inputs."""
    
    def __init__(
        self,
        feature_bounds: Optional[Dict[str, FeatureBounds]] = None,
        anomaly_threshold: float = 3.0,
        max_anomalies: int = 5,
    ):
        """
        Initialize the validator.
        
        Args:
            feature_bounds: Dictionary of feature bounds (uses defaults if not provided)
            anomaly_threshold: Number of std deviations to flag as anomaly
            max_anomalies: Max anomalies before marking as suspicious
        """
        self.feature_bounds = feature_bounds or DEFAULT_FEATURE_BOUNDS
        self.anomaly_threshold = anomaly_threshold
        self.max_anomalies = max_anomalies
    
    def validate(self, transaction: Dict[str, Any]) -> ValidationResult:
        """
        Validate a transaction input.
        
        Args:
            transaction: Dictionary of feature names to values
            
        Returns:
            ValidationResult with validity flag, warnings, and anomaly details
        """
        result = ValidationResult(is_valid=True)
        total_deviation = 0.0
        
        # Check each feature
        for feature, bounds in self.feature_bounds.items():
            if feature not in transaction:
                continue
            
            value = transaction[feature]
            
            # Check for non-numeric values
            if not isinstance(value, (int, float)):
                result.is_valid = False
                result.add_warning(f"{feature}: expected numeric value, got {type(value).__name__}")
                continue
            
            # Check for NaN/Inf
            if np.isnan(value) or np.isinf(value):
                result.is_valid = False
                result.add_warning(f"{feature}: contains NaN or Inf")
                continue
            
            # Check bounds
            deviation = bounds.get_deviation(value)
            
            if deviation > self.anomaly_threshold:
                result.add_anomaly(feature, deviation)
                result.add_warning(
                    f"{feature}: value {value:.4f} is {deviation:.1f}σ from expected "
                    f"(mean={bounds.mean:.4f}, std={bounds.std:.4f})"
                )
            
            total_deviation += deviation
        
        # Calculate overall risk score
        n_features = len([f for f in self.feature_bounds if f in transaction])
        if n_features > 0:
            avg_deviation = total_deviation / n_features
            result.risk_score = min(1.0, avg_deviation / (2 * self.anomaly_threshold))
        
        # Flag as potentially invalid if too many anomalies
        if len(result.anomalies) >= self.max_anomalies:
            result.add_warning(
                f"Transaction has {len(result.anomalies)} anomalous features - "
                "input may be malformed or adversarial"
            )
        
        return result
    
    def validate_batch(
        self,
        transactions: List[Dict[str, Any]],
    ) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """
        Validate a batch of transactions.
        
        Returns:
            Tuple of (individual results, batch summary)
        """
        results = [self.validate(t) for t in transactions]
        
        summary = {
            "total": len(transactions),
            "valid": sum(1 for r in results if r.is_valid),
            "with_warnings": sum(1 for r in results if r.warnings),
            "high_risk": sum(1 for r in results if r.risk_score > 0.5),
            "avg_risk_score": np.mean([r.risk_score for r in results]) if results else 0,
        }
        
        return results, summary
    
    def get_feature_report(self, transaction: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Get detailed report on each feature.
        
        Returns:
            Dictionary with feature analysis
        """
        report = {}
        
        for feature, bounds in self.feature_bounds.items():
            if feature not in transaction:
                continue
            
            value = transaction[feature]
            deviation = bounds.get_deviation(value)
            
            report[feature] = {
                "value": value,
                "mean": bounds.mean,
                "std": bounds.std,
                "deviation_sigma": deviation,
                "within_bounds": bounds.is_within_bounds(value),
                "lower_bound": bounds.lower_bound,
                "upper_bound": bounds.upper_bound,
            }
        
        return report


# Global validator instance
_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator


def validate_transaction(transaction: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate a single transaction."""
    return get_validator().validate(transaction)


def validate_with_details(
    transaction: Dict[str, Any],
) -> Tuple[ValidationResult, Dict[str, Dict]]:
    """Validate and return detailed feature report."""
    validator = get_validator()
    result = validator.validate(transaction)
    report = validator.get_feature_report(transaction)
    return result, report
