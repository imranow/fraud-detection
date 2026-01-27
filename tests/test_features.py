"""Tests for feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.transformers import (
    AmountFeatures,
    AnomalyScoreFeatures,
    InteractionFeatures,
    TimeFeatures,
    VelocityFeatures,
)


class TestAmountFeatures:
    """Tests for AmountFeatures transformer."""

    def test_fit_transform(self, sample_dataframe):
        """Test that amount features are created correctly."""
        transformer = AmountFeatures()
        result = transformer.fit_transform(sample_dataframe)
        
        # Check new columns exist
        assert "Amount_log" in result.columns
        assert "Amount_sqrt" in result.columns
        assert "Amount_bin" in result.columns
        assert "Amount_is_round" in result.columns
        assert "Amount_cents" in result.columns

    def test_log_transform(self, sample_transaction):
        """Test log transform creates valid output."""
        df = pd.DataFrame([sample_transaction])
        
        transformer = AmountFeatures()
        result = transformer.fit_transform(df)
        
        # Log should be finite for positive amount
        assert np.isfinite(result["Amount_log"].iloc[0])
        assert np.isfinite(result["Amount_log1p"].iloc[0])

    def test_round_number_detection(self, sample_transaction):
        """Test round number detection."""
        df = pd.DataFrame([sample_transaction])
        
        # Test with round number
        df["Amount"] = 100.0
        transformer = AmountFeatures()
        result = transformer.fit_transform(df)
        assert result["Amount_is_round"].iloc[0] == 1
        
        # Test with non-round number
        df["Amount"] = 99.99
        result = transformer.fit_transform(df)
        assert result["Amount_is_round"].iloc[0] == 0


class TestTimeFeatures:
    """Tests for TimeFeatures transformer."""

    def test_fit_transform(self, sample_dataframe):
        """Test that time features are created correctly."""
        transformer = TimeFeatures()
        result = transformer.fit_transform(sample_dataframe)
        
        # Check correct column names from implementation
        assert "Hour_of_day" in result.columns
        assert "Time_sin" in result.columns
        assert "Time_cos" in result.columns
        assert "Is_night" in result.columns

    def test_hour_cyclical(self, sample_transaction):
        """Test cyclical hour encoding."""
        transformer = TimeFeatures()
        
        # Test hour 0 (midnight)
        df = pd.DataFrame([sample_transaction])
        df["Time"] = 0.0
        result = transformer.fit_transform(df)
        
        # At hour 0, sin should be 0, cos should be 1
        assert abs(result["Time_sin"].iloc[0]) < 0.01
        assert abs(result["Time_cos"].iloc[0] - 1.0) < 0.01

    def test_night_flag(self, sample_transaction):
        """Test night hour detection."""
        transformer = TimeFeatures()
        df = pd.DataFrame([sample_transaction])
        
        # Test night hour (2 AM = 2*3600 seconds)
        df["Time"] = 2 * 3600
        result = transformer.fit_transform(df)
        assert result["Is_night"].iloc[0] == 1
        
        # Test day hour (2 PM = 14*3600 seconds)
        df["Time"] = 14 * 3600
        result = transformer.fit_transform(df)
        assert result["Is_night"].iloc[0] == 0


class TestVelocityFeatures:
    """Tests for VelocityFeatures transformer."""

    def test_fit_transform(self, sample_dataframe):
        """Test that velocity features are created correctly."""
        transformer = VelocityFeatures()
        result = transformer.fit_transform(sample_dataframe)
        
        assert "V_magnitude" in result.columns
        assert "V_mean" in result.columns
        assert "V_std" in result.columns
        assert "V_max" in result.columns
        assert "V_min" in result.columns
        assert "V_n_outliers" in result.columns

    def test_magnitude_calculation(self, sample_transaction):
        """Test PCA component magnitude calculation."""
        transformer = VelocityFeatures()
        df = pd.DataFrame([sample_transaction])
        result = transformer.fit_transform(df)
        
        # Magnitude should be positive
        assert result["V_magnitude"].iloc[0] > 0

    def test_outlier_detection(self, sample_dataframe):
        """Test outlier count feature."""
        transformer = VelocityFeatures()
        result = transformer.fit_transform(sample_dataframe)
        
        # Outlier count should be non-negative integers
        assert all(result["V_n_outliers"] >= 0)


class TestInteractionFeatures:
    """Tests for InteractionFeatures transformer."""

    def test_fit_transform(self, sample_dataframe):
        """Test that interaction features are created correctly."""
        transformer = InteractionFeatures()
        result = transformer.fit_transform(sample_dataframe)
        
        # Check amount interactions
        assert "Amount_x_V14" in result.columns
        assert "Amount_x_V17" in result.columns
        
        # Check V ratios
        assert "V14_div_V17" in result.columns
        
        # Check squared terms
        assert "V14_squared" in result.columns

    def test_division_handles_zero(self, sample_transaction):
        """Test that division handles zero denominators."""
        df = pd.DataFrame([sample_transaction])
        df["V17"] = 0.0
        
        transformer = InteractionFeatures()
        result = transformer.fit_transform(df)
        
        # Should not have inf values due to epsilon protection
        assert np.isfinite(result["V14_div_V17"].iloc[0])


class TestAnomalyScoreFeatures:
    """Tests for AnomalyScoreFeatures transformer."""

    def test_fit_transform(self, sample_dataframe, sample_labels):
        """Test anomaly score calculation."""
        transformer = AnomalyScoreFeatures()
        result = transformer.fit_transform(sample_dataframe, sample_labels)
        
        assert "Anomaly_zscore_sum" in result.columns
        assert "Anomaly_zscore_max" in result.columns

    def test_transform_consistency(self, sample_dataframe, sample_labels):
        """Test that transform produces consistent results."""
        transformer = AnomalyScoreFeatures()
        transformer.fit(sample_dataframe, sample_labels)
        
        result1 = transformer.transform(sample_dataframe)
        result2 = transformer.transform(sample_dataframe)
        
        pd.testing.assert_frame_equal(result1, result2)
