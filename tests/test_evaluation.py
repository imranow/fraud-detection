"""Tests for evaluation metrics."""

import numpy as np
import pytest

from fraud_detection.evaluation.metrics import EvaluationResult, FraudEvaluator


class TestFraudEvaluator:
    """Tests for FraudEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return FraudEvaluator()

    @pytest.fixture
    def perfect_predictions(self):
        """Perfect predictions for testing."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
        return y_true, y_proba

    @pytest.fixture
    def realistic_predictions(self):
        """Realistic predictions with some errors."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.1, 0.3, 0.6, 0.1, 0.2, 0.1, 0.8, 0.4])
        return y_true, y_proba

    def test_evaluate_perfect(self, evaluator, perfect_predictions):
        """Test evaluation with perfect predictions."""
        y_true, y_proba = perfect_predictions
        result = evaluator.evaluate(y_true, y_proba, threshold=0.5)
        
        assert isinstance(result, EvaluationResult)
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.roc_auc == 1.0

    def test_evaluate_realistic(self, evaluator, realistic_predictions):
        """Test evaluation with realistic predictions."""
        y_true, y_proba = realistic_predictions
        result = evaluator.evaluate(y_true, y_proba, threshold=0.5)
        
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.roc_auc <= 1

    def test_confusion_matrix(self, evaluator, realistic_predictions):
        """Test confusion matrix values."""
        y_true, y_proba = realistic_predictions
        result = evaluator.evaluate(y_true, y_proba, threshold=0.5)
        
        assert result.tn >= 0
        assert result.fp >= 0
        assert result.fn >= 0
        assert result.tp >= 0
        
        # Sum should equal total samples
        total = result.tn + result.fp + result.fn + result.tp
        assert total == len(y_true)

    def test_find_optimal_threshold_f1(self, evaluator, realistic_predictions):
        """Test optimal threshold finding for F1."""
        y_true, y_proba = realistic_predictions
        threshold, score = evaluator.find_optimal_threshold(y_true, y_proba, metric="f1")
        
        assert 0 <= threshold <= 1
        assert 0 <= score <= 1

    def test_find_optimal_threshold_f2(self, evaluator, realistic_predictions):
        """Test optimal threshold finding for F2."""
        y_true, y_proba = realistic_predictions
        threshold, score = evaluator.find_optimal_threshold(y_true, y_proba, metric="f2")
        
        assert 0 <= threshold <= 1

    def test_cost_calculation(self, evaluator, realistic_predictions):
        """Test cost-based evaluation."""
        y_true, y_proba = realistic_predictions
        
        evaluator_custom = FraudEvaluator(cost_fp=10, cost_fn=100)
        result = evaluator_custom.evaluate(y_true, y_proba, threshold=0.5)
        
        expected_cost = result.fp * 10 + result.fn * 100
        assert result.total_cost == expected_cost

    def test_pr_curve_data(self, evaluator, realistic_predictions):
        """Test PR curve data generation."""
        y_true, y_proba = realistic_predictions
        curve_data = evaluator.get_precision_recall_curve(y_true, y_proba)
        
        assert "precision" in curve_data
        assert "recall" in curve_data
        assert len(curve_data["precision"]) == len(curve_data["recall"])

    def test_roc_curve_data(self, evaluator, realistic_predictions):
        """Test ROC curve data generation."""
        y_true, y_proba = realistic_predictions
        curve_data = evaluator.get_roc_curve(y_true, y_proba)
        
        assert "fpr" in curve_data
        assert "tpr" in curve_data
        assert len(curve_data["fpr"]) == len(curve_data["tpr"])


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        result = EvaluationResult(
            accuracy=0.95,
            precision=0.80,
            recall=0.85,
            f1=0.82,
            f2=0.84,
            roc_auc=0.92,
            pr_auc=0.75,
            tn=950,
            fp=20,
            fn=15,
            tp=85,
            threshold=0.5,
            total_cost=350.0,
        )
        
        str_repr = str(result)
        assert "EVALUATION RESULTS" in str_repr
        assert "Accuracy" in str_repr
        assert "0.9500" in str_repr

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = EvaluationResult(
            accuracy=0.95,
            precision=0.80,
            recall=0.85,
            f1=0.82,
            f2=0.84,
            roc_auc=0.92,
            pr_auc=0.75,
            tn=950,
            fp=20,
            fn=15,
            tp=85,
            threshold=0.5,
            total_cost=350.0,
        )
        
        result_dict = result.to_dict()
        assert result_dict["accuracy"] == 0.95
        assert result_dict["precision"] == 0.80
        assert "confusion_matrix" in result_dict
