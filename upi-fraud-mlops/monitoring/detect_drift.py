"""
detect_drift.py
---------------
Pulls production predictions from predictions.db, compares against the
training reference distribution using Evidently, logs PSI per feature,
saves an HTML report, and writes RETRAINING NEEDED alerts to alerts.log.

Also logs PSI scores as MLflow metrics (local file-based tracking).

Usage:
    python monitoring/detect_drift.py
"""

import os
import sys
import sqlite3
import datetime
import joblib
import mlflow
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features import FEATURE_COLUMNS  # noqa: E402

DB_PATH        = os.path.join(PROJECT_ROOT, "data", "predictions.db")
REFERENCE_PATH = os.path.join(PROJECT_ROOT, "data", "reference_data.parquet")
PIPELINE_PATH  = os.path.join(PROJECT_ROOT, "models", "pipeline.pkl")
REPORTS_DIR    = os.path.join(PROJECT_ROOT, "data", "drift_reports")
ALERTS_LOG     = os.path.join(PROJECT_ROOT, "monitoring", "alerts.log")
PSI_THRESHOLD  = 0.2


def load_production(db_path: str, pipeline) -> pd.DataFrame:
    """Load raw rows from DB, run through pipeline to get engineered features."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM predictions", conn)

    # Restore dummy columns required by pipeline transformers
    df["nameOrig"]       = "C000000000"
    df["nameDest"]       = "C000000000"
    df["isFlaggedFraud"] = 0

    X = pipeline.transform(df)
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    return X[FEATURE_COLUMNS]


def compute_psi(reference: pd.Series, production: pd.Series, bins: int = 10) -> float:
    import numpy as np
    ref_counts, edges = pd.cut(reference, bins=bins, retbins=True)
    prod_counts = pd.cut(production, bins=edges)
    ref_pct  = ref_counts.value_counts(normalize=True, sort=False) + 1e-6
    prod_pct = prod_counts.value_counts(normalize=True, sort=False) + 1e-6
    return float(((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)).sum())


def main():
    today = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading reference data ...")
    reference = pd.read_parquet(REFERENCE_PATH)[FEATURE_COLUMNS]

    print("Loading pipeline ...")
    pipeline = joblib.load(PIPELINE_PATH)

    print("Loading production predictions from DB ...")
    production = load_production(DB_PATH, pipeline)
    print(f"  Reference rows : {len(reference):,}")
    print(f"  Production rows: {len(production):,}")

    # ── Evidently drift report ────────────────────────────────────────────────
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=production)

    html_path = os.path.join(REPORTS_DIR, f"drift_{today}.html")
    report.save_html(html_path)
    print(f"HTML report saved -> {html_path}")

    # ── Extract per-feature drift results ─────────────────────────────────────
    result_dict = report.as_dict()
    feature_metrics = result_dict["metrics"][1]["result"]["drift_by_columns"]

    psi_scores = {}
    alert_lines = []
    print(f"\n{'Feature':<25}  {'Drift Score':>12}  {'Drifted?'}")
    print("-" * 55)
    for feat, info in feature_metrics.items():
        score = round(info.get("drift_score", 0.0), 4)
        drifted = info.get("drift_detected", False)
        psi_scores[feat] = score
        status = "[DRIFT]" if drifted else "[stable]"
        print(f"  {feat:<25}  {score:>12.4f}  {status}")

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if score > PSI_THRESHOLD or drifted:
            alert_lines.append(f"{ts} | DRIFT ALERT | feature={feat} | score={score} | RETRAINING NEEDED")
        else:
            alert_lines.append(f"{ts} | STABLE      | feature={feat} | score={score}")

    # ── Write alerts.log ──────────────────────────────────────────────────────
    with open(ALERTS_LOG, "a") as f:
        f.write("\n".join(alert_lines) + "\n")
    print(f"\nAlerts written -> {ALERTS_LOG}")

    # ── MLflow metric logging ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
    mlflow.set_experiment("upi-fraud-drift")
    with mlflow.start_run(run_name=f"drift_{today}"):
        mlflow.log_metrics({f"psi_{k}": v for k, v in psi_scores.items()})
        mlflow.log_artifact(html_path)
    print("MLflow metrics logged.")


if __name__ == "__main__":
    main()
