#!/usr/bin/env python
"""Training script for fraud detection model.

This script provides a complete training pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training with multiple algorithms
- Evaluation and comparison
- Model persistence

Usage:
    python scripts/train.py --model random_forest
    python scripts/train.py --model all --select-features
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fraud_detection.data import FraudDataLoader
from fraud_detection.evaluation import FraudEvaluator, evaluate_model
from fraud_detection.features import FeatureEngineer, FeatureSelector
from fraud_detection.models import get_trainer
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic_regression", "all"],
        help="Model type to train",
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the data file",
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )
    
    parser.add_argument(
        "--feature-engineering",
        action="store_true",
        default=True,
        help="Apply feature engineering",
    )
    
    parser.add_argument(
        "--select-features",
        action="store_true",
        default=False,
        help="Apply feature selection",
    )
    
    parser.add_argument(
        "--n-features",
        type=int,
        default=50,
        help="Number of features to select (if --select-features)",
    )
    
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Save trained model(s)",
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def train_single_model(
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_model: bool = True,
    random_state: int = 42,
) -> Dict:
    """Train and evaluate a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()}")
    logger.info(f"{'='*60}")
    
    # Get trainer
    trainer = get_trainer(model_type, random_state=random_state)
    
    # Train
    trainer.fit(X_train, y_train)
    
    # Predict
    y_proba = trainer.predict_proba(X_test)[:, 1]
    
    # Evaluate
    evaluator = FraudEvaluator()
    
    # Find optimal threshold
    optimal_thresh, best_f2 = evaluator.find_optimal_threshold(
        y_test.values, y_proba, metric="f2"
    )
    logger.info(f"Optimal threshold (F2): {optimal_thresh:.4f}")
    
    # Evaluate at optimal threshold
    result = evaluator.evaluate(y_test.values, y_proba, threshold=optimal_thresh)
    print(result)
    
    # Feature importance
    importance = trainer.get_feature_importance()
    if importance is not None:
        logger.info("\nTop 10 Feature Importances:")
        for feat, imp in importance.head(10).items():
            logger.info(f"  {feat}: {imp:.4f}")
    
    # Save model
    model_path = None
    if save_model:
        model_path = trainer.save()
    
    return {
        "model_type": model_type,
        "trainer": trainer,
        "result": result,
        "optimal_threshold": optimal_thresh,
        "model_path": model_path,
        "feature_importance": importance,
    }


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting fraud detection training pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load data
    logger.info("\n" + "="*60)
    logger.info("LOADING DATA")
    logger.info("="*60)
    
    loader = FraudDataLoader(
        data_path=args.data_path,
        random_state=args.random_state,
    )
    
    X_train, X_test, y_train, y_test = loader.get_train_test_split(
        test_size=args.test_size,
        scale_features=False,  # Will scale after feature engineering
    )
    
    logger.info(f"Train set: {len(X_train):,} samples ({y_train.sum():,} fraud)")
    logger.info(f"Test set: {len(X_test):,} samples ({y_test.sum():,} fraud)")
    
    # Feature engineering
    if args.feature_engineering:
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*60)
        
        fe = FeatureEngineer()
        X_train = fe.fit_transform(X_train, y_train)
        X_test = fe.transform(X_test)
        
        logger.info(f"Features after engineering: {X_train.shape[1]}")
    
    # Feature selection
    if args.select_features:
        logger.info("\n" + "="*60)
        logger.info("FEATURE SELECTION")
        logger.info("="*60)
        
        selector = FeatureSelector(method="importance", n_features=args.n_features)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        
        logger.info(f"Features after selection: {X_train.shape[1]}")
    
    # Train models
    models_to_train = [args.model] if args.model != "all" else [
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
    ]
    
    results = []
    for model_type in models_to_train:
        try:
            result = train_single_model(
                model_type=model_type,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                save_model=args.save_model,
                random_state=args.random_state,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
    
    # Comparison summary
    if len(results) > 1:
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison_data = []
        for r in results:
            comparison_data.append({
                "Model": r["model_type"],
                "ROC-AUC": r["result"].roc_auc,
                "PR-AUC": r["result"].pr_auc,
                "F1": r["result"].f1,
                "F2": r["result"].f2,
                "Precision": r["result"].precision,
                "Recall": r["result"].recall,
                "Threshold": r["optimal_threshold"],
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("F2", ascending=False)
        print("\n" + comparison_df.to_string(index=False))
        
        best_model = comparison_df.iloc[0]["Model"]
        logger.info(f"\n🏆 Best model by F2 score: {best_model}")
    
    logger.info("\n✅ Training complete!")
    
    return results


if __name__ == "__main__":
    main()
