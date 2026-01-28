"""Model definitions and training module."""

from fraud_detection.models.base import BaseTrainer
from fraud_detection.models.trainers import (
    GradientBoostingTrainer,
    LogisticRegressionTrainer,
    RandomForestTrainer,
    get_trainer,
)

__all__ = [
    "BaseTrainer",
    "RandomForestTrainer",
    "GradientBoostingTrainer",
    "LogisticRegressionTrainer",
    "get_trainer",
]

# Add optional trainers if available
try:
    from fraud_detection.models.trainers import XGBoostTrainer  # noqa: F401

    __all__.append("XGBoostTrainer")
except ImportError:
    pass

try:
    from fraud_detection.models.trainers import LightGBMTrainer  # noqa: F401

    __all__.append("LightGBMTrainer")
except ImportError:
    pass
