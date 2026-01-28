"""MLflow configuration and utilities for experiment tracking."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import MLflow, gracefully handle if not installed
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "")


def is_mlflow_available() -> bool:
    """Check if MLflow is available."""
    return MLFLOW_AVAILABLE


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """
    Initialize MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI (default: from env or local)
        experiment_name: Experiment name to use (default: fraud-detection)
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available, skipping setup")
        return
    
    uri = tracking_uri or MLFLOW_TRACKING_URI
    experiment = experiment_name or MLFLOW_EXPERIMENT_NAME
    
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    
    logger.info(f"MLflow initialized: uri={uri}, experiment={experiment}")


@contextmanager
def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
):
    """
    Context manager for MLflow runs.
    
    Falls back to a no-op context if MLflow is not available.
    
    Args:
        run_name: Name for the run
        tags: Tags to add to the run
        nested: Whether this is a nested run
        
    Yields:
        MLflow run object or None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        yield None
        return
    
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
        yield run


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    if not MLFLOW_AVAILABLE:
        return
    
    # MLflow has a limit on param value length, truncate if needed
    truncated_params = {}
    for key, value in params.items():
        str_value = str(value)
        if len(str_value) > 250:
            str_value = str_value[:247] + "..."
        truncated_params[key] = str_value
    
    mlflow.log_params(truncated_params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow."""
    if not MLFLOW_AVAILABLE:
        return
    
    mlflow.log_metrics(metrics, step=step)


def log_model(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """
    Log a scikit-learn model to MLflow.
    
    Args:
        model: The model to log
        artifact_path: Path within the run's artifact directory
        registered_model_name: If provided, register model to MLflow Model Registry
        **kwargs: Additional arguments for mlflow.sklearn.log_model
        
    Returns:
        Model URI if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            **kwargs,
        )
        logger.info(f"Model logged: {model_info.model_uri}")
        return model_info.model_uri
    except Exception as e:
        logger.error(f"Failed to log model: {e}")
        return None


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Log a local file as an artifact."""
    if not MLFLOW_AVAILABLE:
        return
    
    mlflow.log_artifact(local_path, artifact_path)


def log_figure(figure: Any, artifact_file: str) -> None:
    """Log a matplotlib figure as an artifact."""
    if not MLFLOW_AVAILABLE:
        return
    
    mlflow.log_figure(figure, artifact_file)


def set_tags(tags: Dict[str, str]) -> None:
    """Set tags for the current run."""
    if not MLFLOW_AVAILABLE:
        return
    
    mlflow.set_tags(tags)


def get_best_run(
    experiment_name: Optional[str] = None,
    metric: str = "roc_auc",
    ascending: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment (default: current experiment)
        metric: Metric to sort by
        ascending: If True, lower is better; if False, higher is better
        
    Returns:
        Dictionary with run info and metrics, or None if not found
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(
            experiment_name or MLFLOW_EXPERIMENT_NAME
        )
        
        if not experiment:
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        
        if not runs:
            return None
        
        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "artifact_uri": best_run.info.artifact_uri,
        }
    except Exception as e:
        logger.error(f"Failed to get best run: {e}")
        return None


def load_model(model_uri: str) -> Any:
    """
    Load a model from MLflow.
    
    Args:
        model_uri: URI to the model (e.g., "runs:/run_id/model")
        
    Returns:
        Loaded model or None if failed
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_uri}: {e}")
        return None


def register_model(
    model_uri: str,
    name: str,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Register a model from a run to the Model Registry.
    
    Args:
        model_uri: URI to the model
        name: Name for the registered model
        tags: Optional tags for the model version
        
    Returns:
        Model version string or None if failed
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        result = mlflow.register_model(model_uri, name)
        
        if tags:
            client = MlflowClient()
            for key, value in tags.items():
                client.set_model_version_tag(name, result.version, key, value)
        
        logger.info(f"Model registered: {name} version {result.version}")
        return result.version
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None


def transition_model_stage(
    name: str,
    version: str,
    stage: str,
    archive_existing: bool = True,
) -> bool:
    """
    Transition a model version to a new stage.
    
    Args:
        name: Registered model name
        version: Model version
        stage: Target stage (Staging, Production, Archived)
        archive_existing: Whether to archive existing versions in target stage
        
    Returns:
        True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False
    
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Model {name} v{version} transitioned to {stage}")
        return True
    except Exception as e:
        logger.error(f"Failed to transition model stage: {e}")
        return False
