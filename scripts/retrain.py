#!/usr/bin/env python
"""Automated model retraining pipeline for fraud detection."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fraud_detection.data.loader import load_credit_card_data
from fraud_detection.evaluation.metrics import calculate_all_metrics
from fraud_detection.features.engineering import FeatureEngineer
from fraud_detection.models.trainers import RandomForestTrainer
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Configuration
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"
MODELS_PATH = Path(__file__).parent.parent / "models"
METRICS_PATH = Path(__file__).parent.parent / "data" / "processed" / "metrics_history.json"

# Thresholds for automated deployment
MINIMUM_ROC_AUC = 0.95
MINIMUM_RECALL = 0.80
MAXIMUM_PERFORMANCE_DROP = 0.02  # 2% drop from current production


def load_current_production_metrics() -> Optional[Dict[str, float]]:
    """Load metrics from current production model."""
    try:
        # Look for the most recent model's metrics
        model_files = sorted(MODELS_PATH.glob("random_forest_*.joblib"))
        if not model_files:
            return None
        
        latest_model = model_files[-1]
        metrics_file = latest_model.with_suffix(".json")
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                return data.get("metrics", {})
        
        return None
    except Exception as e:
        logger.warning(f"Could not load production metrics: {e}")
        return None


def check_data_drift(
    current_data: pd.DataFrame,
    reference_stats: Optional[Dict] = None,
    threshold: float = 0.1,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check for data drift using statistical tests.
    
    Args:
        current_data: Current dataset
        reference_stats: Reference statistics (from training data)
        threshold: Threshold for detecting drift
        
    Returns:
        Tuple of (drift_detected, drift_scores)
    """
    drift_scores = {}
    
    # If no reference, can't detect drift
    if reference_stats is None:
        return False, {}
    
    # Check numeric features for drift
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in reference_stats:
            ref_mean = reference_stats[col].get("mean", 0)
            ref_std = reference_stats[col].get("std", 1)
            
            current_mean = current_data[col].mean()
            
            # Normalized difference
            if ref_std > 0:
                drift_scores[col] = abs(current_mean - ref_mean) / ref_std
    
    # Check if any feature exceeds threshold
    max_drift = max(drift_scores.values()) if drift_scores else 0
    drift_detected = max_drift > threshold
    
    return drift_detected, drift_scores


def train_new_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[any, Dict[str, float]]:
    """Train a new model and evaluate it."""
    # Initialize trainer
    trainer = RandomForestTrainer(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    
    # Train
    logger.info("Training new model...")
    trainer.train(X_train, y_train)
    
    # Evaluate
    y_pred = trainer.predict(X_val)
    y_proba = trainer.predict_proba(X_val)
    
    metrics = calculate_all_metrics(y_val, y_pred, y_proba)
    
    return trainer, metrics


def should_deploy(
    new_metrics: Dict[str, float],
    current_metrics: Optional[Dict[str, float]],
) -> Tuple[bool, str]:
    """
    Determine if new model should be deployed.
    
    Returns:
        Tuple of (should_deploy, reason)
    """
    # Check minimum thresholds
    if new_metrics.get("roc_auc", 0) < MINIMUM_ROC_AUC:
        return False, f"ROC-AUC {new_metrics.get('roc_auc'):.4f} below minimum {MINIMUM_ROC_AUC}"
    
    if new_metrics.get("recall", 0) < MINIMUM_RECALL:
        return False, f"Recall {new_metrics.get('recall'):.4f} below minimum {MINIMUM_RECALL}"
    
    # If no current model, deploy the new one
    if current_metrics is None:
        return True, "No existing production model"
    
    # Check for performance improvement or acceptable drop
    roc_diff = new_metrics.get("roc_auc", 0) - current_metrics.get("roc_auc", 0)
    recall_diff = new_metrics.get("recall", 0) - current_metrics.get("recall", 0)
    
    if roc_diff < -MAXIMUM_PERFORMANCE_DROP:
        return False, f"ROC-AUC dropped by {abs(roc_diff):.4f}, exceeds {MAXIMUM_PERFORMANCE_DROP}"
    
    if recall_diff < -MAXIMUM_PERFORMANCE_DROP:
        return False, f"Recall dropped by {abs(recall_diff):.4f}, exceeds {MAXIMUM_PERFORMANCE_DROP}"
    
    # Check for improvement
    if roc_diff >= 0 or recall_diff >= 0:
        return True, f"Performance improved (ROC-AUC: {roc_diff:+.4f}, Recall: {recall_diff:+.4f})"
    
    return False, "No significant improvement"


def save_model_and_metrics(
    trainer: any,
    metrics: Dict[str, float],
    feature_names: list,
) -> Path:
    """Save the trained model and its metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = MODELS_PATH / f"random_forest_{timestamp}.joblib"
    trainer.save(model_path)
    
    # Save metrics
    metrics_data = {
        "timestamp": timestamp,
        "metrics": metrics,
        "feature_names": feature_names,
        "model_path": str(model_path),
    }
    
    metrics_file = model_path.with_suffix(".json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"Model saved: {model_path}")
    return model_path


def send_notification(
    message: str,
    success: bool = True,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Send notification about retraining result.
    
    Supports Slack and email via environment variables.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    
    if webhook_url:
        try:
            import requests
            
            color = "#36a64f" if success else "#dc3545"
            payload = {
                "attachments": [{
                    "color": color,
                    "title": "Fraud Detection Model Retraining",
                    "text": message,
                    "fields": [
                        {"title": k, "value": f"{v:.4f}", "short": True}
                        for k, v in (metrics or {}).items()
                    ][:8],  # Limit fields
                }]
            }
            
            requests.post(webhook_url, json=payload, timeout=10)
            logger.info("Slack notification sent")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")
    
    # Log the message regardless
    log_func = logger.info if success else logger.warning
    log_func(f"Retraining result: {message}")


def main():
    """Main retraining pipeline."""
    parser = argparse.ArgumentParser(description="Retrain fraud detection model")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even without data drift",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate but don't deploy",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to training data",
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION MODEL RETRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Force retrain: {args.force}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Load data
    logger.info("\n1. Loading data...")
    if not args.data_path.exists():
        logger.error(f"Data not found: {args.data_path}")
        send_notification("Retraining failed: data not found", success=False)
        return 1
    
    df = load_credit_card_data(args.data_path)
    logger.info(f"   Loaded {len(df):,} transactions")
    
    # Check for data drift (if not forcing)
    if not args.force:
        logger.info("\n2. Checking for data drift...")
        drift_detected, drift_scores = check_data_drift(df)
        
        if not drift_detected:
            logger.info("   No significant data drift detected")
            logger.info("   Use --force to retrain anyway")
            return 0
        else:
            logger.info(f"   Data drift detected! Max drift score: {max(drift_scores.values()):.4f}")
    
    # Feature engineering
    logger.info("\n3. Feature engineering...")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    engineer = FeatureEngineer()
    X_engineered = engineer.fit_transform(X)
    logger.info(f"   Features: {X.shape[1]} -> {X_engineered.shape[1]}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"   Train: {len(X_train):,}, Validation: {len(X_val):,}")
    
    # Train new model
    logger.info("\n4. Training model...")
    trainer, new_metrics = train_new_model(X_train, y_train, X_val, y_val)
    
    logger.info("\n   New model metrics:")
    for metric, value in sorted(new_metrics.items()):
        logger.info(f"     {metric}: {value:.4f}")
    
    # Compare with current production
    logger.info("\n5. Comparing with production model...")
    current_metrics = load_current_production_metrics()
    
    if current_metrics:
        logger.info("   Current production metrics:")
        for metric, value in sorted(current_metrics.items()):
            if isinstance(value, float):
                logger.info(f"     {metric}: {value:.4f}")
    else:
        logger.info("   No production model found")
    
    # Decide on deployment
    should_deploy_flag, reason = should_deploy(new_metrics, current_metrics)
    logger.info(f"\n6. Deployment decision: {should_deploy_flag}")
    logger.info(f"   Reason: {reason}")
    
    if args.dry_run:
        logger.info("\n   [DRY RUN] Skipping deployment")
        send_notification(
            f"Dry run complete. Would deploy: {should_deploy_flag}. {reason}",
            success=True,
            metrics=new_metrics,
        )
        return 0
    
    if should_deploy_flag:
        # Save and deploy
        logger.info("\n7. Deploying new model...")
        model_path = save_model_and_metrics(
            trainer,
            new_metrics,
            list(X_engineered.columns),
        )
        
        send_notification(
            f"New model deployed: {model_path.name}. {reason}",
            success=True,
            metrics=new_metrics,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("RETRAINING COMPLETE - MODEL DEPLOYED")
        logger.info("=" * 60)
        return 0
    else:
        send_notification(
            f"Model not deployed. {reason}",
            success=False,
            metrics=new_metrics,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("RETRAINING COMPLETE - MODEL NOT DEPLOYED")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
