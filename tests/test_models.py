"""Tests for model trainers."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.trainers import (
    RandomForestTrainer,
    GradientBoostingTrainer,
    LogisticRegressionTrainer,
)


class TestRandomForestTrainer:
    """Tests for RandomForestTrainer."""

    def test_initialization(self):
        """Test trainer initialization with defaults."""
        trainer = RandomForestTrainer()
        assert trainer.model_name == "random_forest"

    def test_fit_predict(self, sample_dataframe, sample_labels):
        """Test training and prediction."""
        trainer = RandomForestTrainer(n_estimators=10, random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        predictions = trainer.predict(sample_dataframe)
        assert len(predictions) == len(sample_dataframe)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_dataframe, sample_labels):
        """Test probability predictions."""
        trainer = RandomForestTrainer(n_estimators=10, random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        probas = trainer.predict_proba(sample_dataframe)
        assert probas.shape == (len(sample_dataframe), 2)
        assert all(0 <= p <= 1 for p in probas[:, 1])

    def test_feature_importance(self, sample_dataframe, sample_labels):
        """Test feature importance extraction."""
        trainer = RandomForestTrainer(n_estimators=10, random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        importance = trainer.get_feature_importance()
        assert importance is not None
        assert len(importance) == len(sample_dataframe.columns)
        assert abs(importance.sum() - 1.0) < 0.01  # Should sum to ~1


class TestGradientBoostingTrainer:
    """Tests for GradientBoostingTrainer."""

    def test_fit_predict(self, sample_dataframe, sample_labels):
        """Test training and prediction."""
        trainer = GradientBoostingTrainer(n_estimators=10, random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        predictions = trainer.predict(sample_dataframe)
        assert len(predictions) == len(sample_dataframe)

    def test_predict_proba(self, sample_dataframe, sample_labels):
        """Test probability predictions."""
        trainer = GradientBoostingTrainer(n_estimators=10, random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        probas = trainer.predict_proba(sample_dataframe)
        assert probas.shape[1] == 2


class TestLogisticRegressionTrainer:
    """Tests for LogisticRegressionTrainer."""

    def test_fit_predict(self, sample_dataframe, sample_labels):
        """Test training and prediction."""
        trainer = LogisticRegressionTrainer(random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        predictions = trainer.predict(sample_dataframe)
        assert len(predictions) == len(sample_dataframe)

    def test_predict_proba(self, sample_dataframe, sample_labels):
        """Test probability predictions."""
        trainer = LogisticRegressionTrainer(random_state=42)
        trainer.fit(sample_dataframe, sample_labels)
        
        probas = trainer.predict_proba(sample_dataframe)
        assert probas.shape == (len(sample_dataframe), 2)
