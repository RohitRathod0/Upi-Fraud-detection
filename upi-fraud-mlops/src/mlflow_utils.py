"""
mlflow_utils.py
---------------
Helper module for MLflow setup and experiment logging.
Not a runnable script — imported by log_existing_run.py and train.py.
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRACKING_URI  = f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"
REGISTRY_NAME = "upi-fraud-lgbm"


def setup_mlflow(experiment_name: str) -> None:
    """Set SQLite tracking URI and create experiment if it does not exist."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)


def log_training_run(
    params: dict,
    metrics: dict,
    model_path: Path,
    pipeline_path: Path,
    experiment_name: str = "upi-fraud-detection",
) -> str:
    """
    Log a training run with params, metrics, and artifacts; register model.

    Returns the MLflow run_id.
    """
    setup_mlflow(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path),    artifact_path="model_artifacts")
        mlflow.log_artifact(str(pipeline_path), artifact_path="model_artifacts")

        # Register via Model Registry
        model_uri = f"runs:/{run.info.run_id}/model_artifacts"
        try:
            mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
        except mlflow.exceptions.MlflowException:
            # Registry entry already exists — version bump handled automatically
            pass

        run_id = run.info.run_id

    return run_id


def get_latest_run(experiment_name: str = "upi-fraud-detection") -> dict:
    """Return params and metrics of the most recent run in the experiment."""
    setup_mlflow(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {}

    latest = runs[0]
    return {
        "run_id":  latest.info.run_id,
        "params":  latest.data.params,
        "metrics": latest.data.metrics,
        "status":  latest.info.status,
    }
