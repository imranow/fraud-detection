"""A/B Testing infrastructure for model comparison."""

import hashlib
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class TrafficSplit(Enum):
    """Traffic split strategies."""
    
    RANDOM = "random"           # Random assignment
    HASH_BASED = "hash_based"   # Deterministic based on transaction hash
    TIME_BASED = "time_based"   # Alternate over time periods


@dataclass
class ExperimentVariant:
    """A variant in an A/B experiment."""
    
    name: str
    weight: float  # Fraction of traffic (0.0 to 1.0)
    model_path: Optional[str] = None
    model: Any = None
    
    # Metrics tracking
    total_predictions: int = 0
    fraud_predictions: int = 0
    total_latency_ms: float = 0.0
    
    # For statistical analysis
    predictions: List[float] = field(default_factory=list)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.total_predictions
    
    @property
    def fraud_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.fraud_predictions / self.total_predictions
    
    def record_prediction(
        self,
        is_fraud: bool,
        probability: float,
        latency_ms: float,
    ) -> None:
        """Record a prediction for this variant."""
        self.total_predictions += 1
        if is_fraud:
            self.fraud_predictions += 1
        self.total_latency_ms += latency_ms
        self.predictions.append(probability)


@dataclass
class Experiment:
    """An A/B testing experiment."""
    
    name: str
    variants: List[ExperimentVariant]
    split_strategy: TrafficSplit = TrafficSplit.HASH_BASED
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self):
        # Validate weights sum to 1.0
        total_weight = sum(v.weight for v in self.variants)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
    
    @property
    def control(self) -> ExperimentVariant:
        """Get the control variant (first one)."""
        return self.variants[0]
    
    @property
    def treatment(self) -> Optional[ExperimentVariant]:
        """Get the treatment variant (second one, if exists)."""
        if len(self.variants) > 1:
            return self.variants[1]
        return None
    
    def get_variant(self, transaction_id: str) -> ExperimentVariant:
        """
        Get the variant for a given transaction.
        
        Uses the configured split strategy to determine assignment.
        """
        if not self.is_active:
            return self.control
        
        if self.split_strategy == TrafficSplit.RANDOM:
            return self._random_split()
        elif self.split_strategy == TrafficSplit.HASH_BASED:
            return self._hash_split(transaction_id)
        else:
            return self._time_split()
    
    def _random_split(self) -> ExperimentVariant:
        """Random variant assignment."""
        r = random.random()
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant
        return self.variants[-1]
    
    def _hash_split(self, transaction_id: str) -> ExperimentVariant:
        """Deterministic hash-based assignment."""
        # Create a hash of the transaction ID
        hash_val = int(hashlib.md5(transaction_id.encode()).hexdigest(), 16)
        bucket = (hash_val % 1000) / 1000.0
        
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if bucket <= cumulative:
                return variant
        return self.variants[-1]
    
    def _time_split(self) -> ExperimentVariant:
        """Time-based alternation."""
        # Switch variant every minute
        minute = datetime.now().minute
        idx = minute % len(self.variants)
        return self.variants[idx]
    
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results summary."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "variants": [
                {
                    "name": v.name,
                    "weight": v.weight,
                    "total_predictions": v.total_predictions,
                    "fraud_rate": v.fraud_rate,
                    "avg_latency_ms": v.avg_latency_ms,
                }
                for v in self.variants
            ],
            "statistical_significance": self._calculate_significance(),
        }
    
    def _calculate_significance(self) -> Dict[str, Any]:
        """Calculate statistical significance between control and treatment."""
        if not self.treatment or len(self.control.predictions) < 30:
            return {"sufficient_data": False}
        
        control_preds = np.array(self.control.predictions)
        treatment_preds = np.array(self.treatment.predictions)
        
        if len(treatment_preds) < 30:
            return {"sufficient_data": False}
        
        # Two-sample t-test approximation
        n1, n2 = len(control_preds), len(treatment_preds)
        mean1, mean2 = control_preds.mean(), treatment_preds.mean()
        std1, std2 = control_preds.std(), treatment_preds.std()
        
        # Pooled standard error
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        
        if se == 0:
            return {"sufficient_data": True, "significant": False, "reason": "No variance"}
        
        # T-statistic
        t_stat = (mean2 - mean1) / se
        
        # Approximate p-value (two-tailed)
        # Using rough critical values: |t| > 1.96 -> p < 0.05
        is_significant = abs(t_stat) > 1.96
        
        return {
            "sufficient_data": True,
            "control_mean": float(mean1),
            "treatment_mean": float(mean2),
            "difference": float(mean2 - mean1),
            "t_statistic": float(t_stat),
            "significant_at_95": is_significant,
        }


class ABTestManager:
    """Manage A/B testing experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiment: Optional[str] = None
    
    def create_experiment(
        self,
        name: str,
        control_name: str = "control",
        treatment_name: str = "treatment",
        treatment_weight: float = 0.5,
        split_strategy: TrafficSplit = TrafficSplit.HASH_BASED,
    ) -> Experiment:
        """
        Create a new A/B experiment.
        
        Args:
            name: Unique experiment name
            control_name: Name for control variant
            treatment_name: Name for treatment variant
            treatment_weight: Fraction of traffic for treatment (0.0-1.0)
            split_strategy: How to split traffic
            
        Returns:
            Created Experiment
        """
        variants = [
            ExperimentVariant(name=control_name, weight=1.0 - treatment_weight),
            ExperimentVariant(name=treatment_name, weight=treatment_weight),
        ]
        
        experiment = Experiment(
            name=name,
            variants=variants,
            split_strategy=split_strategy,
        )
        
        self.experiments[name] = experiment
        logger.info(f"Created experiment: {name} with {len(variants)} variants")
        
        return experiment
    
    def start_experiment(self, name: str) -> bool:
        """Activate an experiment."""
        if name not in self.experiments:
            logger.warning(f"Experiment not found: {name}")
            return False
        
        # Deactivate current active experiment
        if self.active_experiment:
            self.stop_experiment(self.active_experiment)
        
        self.experiments[name].is_active = True
        self.active_experiment = name
        logger.info(f"Started experiment: {name}")
        
        return True
    
    def stop_experiment(self, name: str) -> Optional[Dict]:
        """
        Stop an experiment and return results.
        
        Returns:
            Experiment results or None if not found
        """
        if name not in self.experiments:
            return None
        
        experiment = self.experiments[name]
        experiment.is_active = False
        experiment.end_time = datetime.now()
        
        if self.active_experiment == name:
            self.active_experiment = None
        
        results = experiment.get_results()
        logger.info(f"Stopped experiment: {name}")
        
        return results
    
    def get_variant(self, transaction_id: str) -> Tuple[Optional[str], Optional[ExperimentVariant]]:
        """
        Get the variant for a transaction in the active experiment.
        
        Returns:
            Tuple of (experiment_name, variant) or (None, None) if no active experiment
        """
        if not self.active_experiment:
            return None, None
        
        experiment = self.experiments[self.active_experiment]
        variant = experiment.get_variant(transaction_id)
        
        return self.active_experiment, variant
    
    def record_prediction(
        self,
        experiment_name: str,
        variant_name: str,
        is_fraud: bool,
        probability: float,
        latency_ms: float,
    ) -> None:
        """Record a prediction result for an experiment variant."""
        if experiment_name not in self.experiments:
            return
        
        experiment = self.experiments[experiment_name]
        for variant in experiment.variants:
            if variant.name == variant_name:
                variant.record_prediction(is_fraud, probability, latency_ms)
                break
    
    def get_all_results(self) -> List[Dict]:
        """Get results for all experiments."""
        return [exp.get_results() for exp in self.experiments.values()]
    
    def get_active_experiment(self) -> Optional[Dict]:
        """Get the currently active experiment info."""
        if not self.active_experiment:
            return None
        return self.experiments[self.active_experiment].get_results()


# Global manager instance
_manager: Optional[ABTestManager] = None


def get_ab_manager() -> ABTestManager:
    """Get or create the global A/B test manager."""
    global _manager
    if _manager is None:
        _manager = ABTestManager()
    return _manager


def is_ab_testing_enabled() -> bool:
    """Check if A/B testing is enabled via environment variable."""
    return os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
