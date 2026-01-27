"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_transaction() -> dict:
    """Single sample transaction for testing."""
    return {
        "Time": 0.0,
        "V1": -1.359807134,
        "V2": -0.072781173,
        "V3": 2.536346738,
        "V4": 1.378155224,
        "V5": -0.338321107,
        "V6": 0.462387778,
        "V7": 0.239598554,
        "V8": 0.098697901,
        "V9": 0.363787079,
        "V10": 0.090794172,
        "V11": -0.551599533,
        "V12": -0.617800856,
        "V13": -0.991389847,
        "V14": -0.311169354,
        "V15": 1.468176972,
        "V16": -0.470400525,
        "V17": 0.207971242,
        "V18": 0.02579058,
        "V19": 0.403992960,
        "V20": 0.251412098,
        "V21": -0.018306778,
        "V22": 0.277837576,
        "V23": -0.110473910,
        "V24": 0.066928075,
        "V25": 0.128539358,
        "V26": -0.189114844,
        "V27": 0.133558377,
        "V28": -0.021053053,
        "Amount": 149.62,
    }


@pytest.fixture
def sample_dataframe(sample_transaction) -> pd.DataFrame:
    """Sample DataFrame with multiple transactions."""
    # Create 100 sample transactions with some variation
    np.random.seed(42)
    n_samples = 100
    
    data = []
    for i in range(n_samples):
        tx = sample_transaction.copy()
        # Add noise to create variation
        for key in tx:
            if key != "Time":
                tx[key] = tx[key] + np.random.normal(0, 0.1)
        tx["Time"] = float(i * 100)
        data.append(tx)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels() -> pd.Series:
    """Sample labels for testing (5% fraud rate)."""
    np.random.seed(42)
    labels = np.zeros(100)
    labels[:5] = 1  # 5 fraud cases
    np.random.shuffle(labels)
    return pd.Series(labels, name="Class")


@pytest.fixture
def fraud_transaction() -> dict:
    """Sample fraudulent transaction (based on fraud patterns)."""
    return {
        "Time": 472.0,
        "V1": -2.3015,
        "V2": 1.7598,
        "V3": -0.3598,
        "V4": 2.3304,
        "V5": -0.8213,
        "V6": -0.0758,
        "V7": 0.5627,
        "V8": -0.3999,
        "V9": -0.2381,
        "V10": -1.5253,
        "V11": 2.0325,
        "V12": -4.8312,
        "V13": 0.2145,
        "V14": -6.5871,  # V14 is very negative - fraud signal
        "V15": -0.2376,
        "V16": -1.3088,
        "V17": -5.5401,
        "V18": -1.8095,
        "V19": 0.7684,
        "V20": 0.4052,
        "V21": 0.3989,
        "V22": 0.8151,
        "V23": 0.0387,
        "V24": -0.4252,
        "V25": -0.0698,
        "V26": -0.0237,
        "V27": 0.5995,
        "V28": 0.3032,
        "Amount": 1.00,
    }
