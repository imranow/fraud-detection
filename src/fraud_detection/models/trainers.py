"""Fraud detection model implementations.

Provides concrete model trainers for various algorithms including
ensemble methods optimized for imbalanced classification.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from fraud_detection.models.base import BaseTrainer
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


# Check for optional dependencies
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError as e:
    HAS_XGBOOST = False
    logger.warning(f"XGBoost not available: {type(e).__name__}")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError as e:
    HAS_LIGHTGBM = False
    logger.warning(f"LightGBM not available: {type(e).__name__}")


class RandomForestTrainer(BaseTrainer):
    """Random Forest trainer for fraud detection."""

    def __init__(
        self,
        random_state: int = 42,
        model_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            model_name="random_forest",
            random_state=random_state,
            model_dir=model_dir,
        )
        self._model_kwargs = kwargs

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def _create_model(self, **kwargs) -> RandomForestClassifier:
        params = self.get_default_params()
        params.update(self._model_kwargs)
        params.update(kwargs)
        return RandomForestClassifier(**params)


class GradientBoostingTrainer(BaseTrainer):
    """Gradient Boosting trainer for fraud detection."""

    def __init__(
        self,
        random_state: int = 42,
        model_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            model_name="gradient_boosting",
            random_state=random_state,
            model_dir=model_dir,
        )
        self._model_kwargs = kwargs

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "min_samples_split": 10,
            "random_state": self.random_state,
        }

    def _create_model(self, **kwargs) -> GradientBoostingClassifier:
        params = self.get_default_params()
        params.update(self._model_kwargs)
        params.update(kwargs)
        return GradientBoostingClassifier(**params)


class LogisticRegressionTrainer(BaseTrainer):
    """Logistic Regression trainer for fraud detection."""

    def __init__(
        self,
        random_state: int = 42,
        model_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            model_name="logistic_regression",
            random_state=random_state,
            model_dir=model_dir,
        )
        self._model_kwargs = kwargs

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def _create_model(self, **kwargs) -> LogisticRegression:
        params = self.get_default_params()
        params.update(self._model_kwargs)
        params.update(kwargs)
        return LogisticRegression(**params)


class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer for fraud detection."""

    def __init__(
        self,
        random_state: int = 42,
        model_dir: Optional[Path] = None,
        use_early_stopping: bool = True,
        early_stopping_rounds: int = 50,
        **kwargs,
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        super().__init__(
            model_name="xgboost",
            random_state=random_state,
            model_dir=model_dir,
        )
        self._model_kwargs = kwargs
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,  # Will be calculated based on class imbalance
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def _create_model(self, **kwargs) -> "xgb.XGBClassifier":
        params = self.get_default_params()
        params.update(self._model_kwargs)
        params.update(kwargs)

        if self.use_early_stopping:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        return xgb.XGBClassifier(**params)

    def _prepare_fit_kwargs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        fit_kwargs = {}

        if self.use_early_stopping and eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False

        return fit_kwargs


class LightGBMTrainer(BaseTrainer):
    """LightGBM trainer for fraud detection."""

    def __init__(
        self,
        random_state: int = 42,
        model_dir: Optional[Path] = None,
        use_early_stopping: bool = True,
        early_stopping_rounds: int = 50,
        **kwargs,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

        super().__init__(
            model_name="lightgbm",
            random_state=random_state,
            model_dir=model_dir,
        )
        self._model_kwargs = kwargs
        self.use_early_stopping = use_early_stopping
        self.early_stopping_rounds = early_stopping_rounds

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 500,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0,
            "reg_lambda": 0,
            "is_unbalance": True,
            "objective": "binary",
            "metric": "auc",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

    def _create_model(self, **kwargs) -> "lgb.LGBMClassifier":
        params = self.get_default_params()
        params.update(self._model_kwargs)
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)

    def _prepare_fit_kwargs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        fit_kwargs = {}

        if self.use_early_stopping and eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        return fit_kwargs


def get_trainer(
    model_type: str = "random_forest",
    random_state: int = 42,
    **kwargs,
) -> BaseTrainer:
    """
    Factory function to get a trainer by model type.

    Args:
        model_type: One of 'random_forest', 'gradient_boosting',
                   'logistic_regression', 'xgboost', 'lightgbm'
        random_state: Random seed
        **kwargs: Additional model parameters

    Returns:
        Configured trainer instance
    """
    trainers = {
        "random_forest": RandomForestTrainer,
        "gradient_boosting": GradientBoostingTrainer,
        "logistic_regression": LogisticRegressionTrainer,
    }

    if HAS_XGBOOST:
        trainers["xgboost"] = XGBoostTrainer

    if HAS_LIGHTGBM:
        trainers["lightgbm"] = LightGBMTrainer

    if model_type not in trainers:
        available = list(trainers.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    return trainers[model_type](random_state=random_state, **kwargs)
