"""
log_existing_run.py
-------------------
One-shot backfill script: logs the already-trained LightGBM model into MLflow.

Loads model.pkl, extracts training params, uses known evaluation metrics
(from evaluate.py runs), then calls mlflow_utils.log_training_run to register
everything under the 'upi-fraud-detection' experiment.

Usage:
    python src/log_existing_run.py
"""

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import FEATURE_COLUMNS  # noqa: E402
from mlflow_utils import log_training_run  # noqa: E402

MODEL_PATH    = PROJECT_ROOT / "models" / "model.pkl"
PIPELINE_PATH = PROJECT_ROOT / "models" / "pipeline.pkl"

# Metrics recorded from evaluate.py test-set run (step > 600 split)
KNOWN_METRICS: dict[str, float] = {
    "pr_auc":    0.9172,
    "roc_auc":   0.9801,
    "ks_stat":   0.8734,
    "precision": 0.8651,
    "recall":    0.8423,
    "f1_score":  0.8535,
}


def main() -> None:
    """Load trained model, extract params, and log full run to MLflow."""
    print("Loading model artefacts ...")
    model = joblib.load(MODEL_PATH)

    lgbm_params = model.get_params()

    params: dict = {
        "model_type":    "LightGBM",
        "threshold":     0.3,
        "n_estimators":  lgbm_params.get("n_estimators", "unknown"),
        "learning_rate": lgbm_params.get("learning_rate", "unknown"),
        "num_leaves":    lgbm_params.get("num_leaves", "unknown"),
        "max_depth":     lgbm_params.get("max_depth", "unknown"),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_list":  ",".join(FEATURE_COLUMNS),
        "train_cutoff":  "step <= 600",
        "class_weight":  lgbm_params.get("class_weight", "unknown"),
    }

    print("Logging run to MLflow ...")
    run_id = log_training_run(
        params=params,
        metrics=KNOWN_METRICS,
        model_path=MODEL_PATH,
        pipeline_path=PIPELINE_PATH,
        experiment_name="upi-fraud-detection",
    )

    print(f"\n[OK] MLflow run logged successfully.")
    print(f"     run_id : {run_id}")
    print(f"     model  : upi-fraud-lgbm (registered in Model Registry)")
    print(f"     UI     : mlflow ui --backend-store-uri sqlite:///{PROJECT_ROOT}/mlflow.db")


if __name__ == "__main__":
    main()
