"""Evaluation metrics for fraud detection models.

Provides comprehensive metrics optimized for imbalanced classification,
including precision-recall curves, confusion matrices, and cost-based metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float  # F2 score (weights recall higher than precision)

    # AUC metrics
    roc_auc: float
    pr_auc: float  # Precision-Recall AUC

    # Confusion matrix
    tn: int
    fp: int
    fn: int
    tp: int

    # Threshold-specific
    threshold: float

    # Optional: cost-based metrics
    total_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": {
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn,
                "tp": self.tp,
            },
            "threshold": self.threshold,
            "total_cost": self.total_cost,
        }

    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1 Score:  {self.f1:.4f}",
            f"F2 Score:  {self.f2:.4f}",
            f"ROC AUC:   {self.roc_auc:.4f}",
            f"PR AUC:    {self.pr_auc:.4f}",
            "-" * 50,
            "Confusion Matrix:",
            f"  TN: {self.tn:,}  FP: {self.fp:,}",
            f"  FN: {self.fn:,}  TP: {self.tp:,}",
            f"Threshold: {self.threshold:.4f}",
        ]
        if self.total_cost is not None:
            lines.append(f"Total Cost: ${self.total_cost:,.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)


class FraudEvaluator:
    """Comprehensive evaluator for fraud detection models.

    Provides metrics optimized for highly imbalanced datasets where
    false negatives (missed fraud) are typically more costly than
    false positives (false alarms).
    """

    def __init__(
        self,
        cost_fp: float = 10.0,
        cost_fn: float = 500.0,
        default_threshold: float = 0.5,
    ):
        """
        Initialize the evaluator.

        Args:
            cost_fp: Cost of a false positive (false alarm)
            cost_fn: Cost of a false negative (missed fraud)
            default_threshold: Default classification threshold
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.default_threshold = default_threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: Optional[float] = None,
    ) -> EvaluationResult:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            threshold: Classification threshold (default: 0.5)

        Returns:
            EvaluationResult with all metrics
        """
        threshold = threshold or self.default_threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate cost
        total_cost = (fp * self.cost_fp) + (fn * self.cost_fn)

        # AUC scores
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        return EvaluationResult(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
            threshold=threshold,
            total_cost=total_cost,
        )

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'f2', 'recall', 'cost')

        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = -np.inf if metric != "cost" else np.inf
        best_threshold = 0.5

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            elif metric == "f2":
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            elif metric == "cost":
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                score = (fp * self.cost_fp) + (fn * self.cost_fn)
                if score < best_score:
                    best_score = score
                    best_threshold = thresh

        return best_threshold, best_score

    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get precision-recall curve data.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with precision, recall, and thresholds
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }

    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary with fpr, tpr, and thresholds
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        """Get sklearn classification report."""
        return str(
            classification_report(
                y_true,
                y_pred,
                target_names=["Non-Fraud", "Fraud"],
            )
        )

    def evaluate_at_multiple_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate at multiple thresholds.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        results = []
        for thresh in thresholds:
            result = self.evaluate(y_true, y_proba, threshold=thresh)
            results.append(result.to_dict())

        return pd.DataFrame(results)


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    print_results: bool = True,
) -> EvaluationResult:
    """
    Convenience function to evaluate model predictions.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Classification threshold
        print_results: Whether to print results

    Returns:
        EvaluationResult
    """
    evaluator = FraudEvaluator()
    result = evaluator.evaluate(y_true, y_proba, threshold)

    if print_results:
        print(result)

    return result
