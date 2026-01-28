"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class FraudDataLoader:
    """Load and preprocess fraud detection dataset."""

    FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    TARGET_COLUMN = "Class"

    def __init__(
        self,
        data_path: Optional[Path] = None,
        random_state: int = 42,
    ):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the CSV file. Defaults to data/raw/creditcard.csv
            random_state: Random seed for reproducibility
        """
        if data_path is None:
            data_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "raw"
                / "creditcard.csv"
            )

        self.data_path = Path(data_path)
        self.random_state = random_state
        self._df: Optional[pd.DataFrame] = None
        self._scaler: Optional[StandardScaler] = None

    def load(self) -> pd.DataFrame:
        """Load the dataset from CSV."""
        if self._df is None:
            logger.info(f"Loading data from {self.data_path}")
            self._df = pd.read_csv(self.data_path)
            logger.info(
                f"Loaded {len(self._df):,} records with {len(self._df.columns)} columns"
            )
        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Get the loaded dataframe."""
        if self._df is None:
            self.load()
        return self._df

    def get_features_and_target(
        self,
        scale_features: bool = True,
        exclude_time: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and target vector.

        Args:
            scale_features: Whether to standardize Amount (and Time if included)
            exclude_time: Whether to exclude Time from features

        Returns:
            Tuple of (X, y)
        """
        df = self.df.copy()

        # Select feature columns
        feature_cols = self.FEATURE_COLUMNS.copy()
        if exclude_time:
            feature_cols.remove("Time")

        X = df[feature_cols].copy()
        y = df[self.TARGET_COLUMN].copy()

        if scale_features:
            # V1-V28 are already scaled via PCA, only scale Amount and Time
            cols_to_scale = ["Amount"]
            if not exclude_time:
                cols_to_scale.append("Time")

            self._scaler = StandardScaler()
            X[cols_to_scale] = self._scaler.fit_transform(X[cols_to_scale])
            logger.info(f"Scaled columns: {cols_to_scale}")

        return X, y

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        scale_features: bool = True,
        exclude_time: bool = False,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get train/test split of the data.

        Args:
            test_size: Proportion of data for testing
            scale_features: Whether to standardize Amount and Time
            exclude_time: Whether to exclude Time from features
            stratify: Whether to stratify by target class

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = self.get_features_and_target(
            scale_features=scale_features,
            exclude_time=exclude_time,
        )

        stratify_col = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col,
        )

        logger.info(
            f"Train set: {len(X_train):,} samples "
            f"(Fraud: {y_train.sum():,}, {y_train.mean()*100:.3f}%)"
        )
        logger.info(
            f"Test set:  {len(X_test):,} samples "
            f"(Fraud: {y_test.sum():,}, {y_test.mean()*100:.3f}%)"
        )

        return X_train, X_test, y_train, y_test

    def get_class_weights(self) -> dict:
        """
        Calculate class weights for handling imbalance.

        Returns:
            Dictionary with class weights {0: weight, 1: weight}
        """
        y = self.df[self.TARGET_COLUMN]
        n_samples = len(y)
        n_classes = 2

        class_counts = y.value_counts()
        weights = {
            cls: n_samples / (n_classes * count) for cls, count in class_counts.items()
        }

        logger.info(f"Class weights: {weights}")
        return weights

    def get_summary(self) -> dict:
        """Get a summary of the dataset."""
        df = self.df

        fraud_count = df[self.TARGET_COLUMN].sum()
        non_fraud_count = len(df) - fraud_count

        return {
            "total_records": len(df),
            "n_features": len(self.FEATURE_COLUMNS),
            "fraud_count": int(fraud_count),
            "non_fraud_count": int(non_fraud_count),
            "fraud_rate": fraud_count / len(df),
            "imbalance_ratio": non_fraud_count / fraud_count,
            "amount_stats": {
                "mean": df["Amount"].mean(),
                "std": df["Amount"].std(),
                "min": df["Amount"].min(),
                "max": df["Amount"].max(),
            },
        }


def load_data(
    data_path: Optional[Path] = None,
    test_size: float = 0.2,
    scale_features: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to load and split data.

    Args:
        data_path: Path to the CSV file
        test_size: Proportion of data for testing
        scale_features: Whether to standardize Amount and Time
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    loader = FraudDataLoader(data_path=data_path, random_state=random_state)
    return loader.get_train_test_split(
        test_size=test_size,
        scale_features=scale_features,
    )
