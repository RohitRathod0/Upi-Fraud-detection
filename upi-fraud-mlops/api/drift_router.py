"""
drift_router.py
---------------
FastAPI APIRouter for Evidently-based drift detection endpoints.

Endpoints:
    GET /drift/drift-report        — JSON drift summary (cached 1 hr)
    GET /drift/drift-report/html   — FileResponse of latest HTML report
    GET /drift/drift-report/status — lightweight status JSON
"""

import os
import sqlite3
import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DB_PATH         = PROJECT_ROOT / "data" / "predictions.db"
REFERENCE_PATH  = PROJECT_ROOT / "data" / "reference_data.parquet"
REPORTS_DIR     = PROJECT_ROOT / "data" / "drift_reports"
PIPELINE_PATH   = PROJECT_ROOT / "models" / "pipeline.pkl"
CACHE_TTL_SECS  = 3600        # re-run Evidently at most once per hour
RECENT_ROWS     = 500         # rows pulled from predictions.db per run

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from features import FEATURE_COLUMNS  # noqa: E402

# ── Module-level state ────────────────────────────────────────────────────────
_cache: dict[str, Any] = {}   # keys: result, timestamp, html_path, alert_count

# ── Load reference data once ──────────────────────────────────────────────────
_reference_df: pd.DataFrame = pd.read_parquet(REFERENCE_PATH)[FEATURE_COLUMNS]
_pipeline = joblib.load(PIPELINE_PATH)

router = APIRouter()


def _load_recent_production() -> pd.DataFrame:
    """Pull last RECENT_ROWS raw rows from predictions.db and apply feature pipeline."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"SELECT * FROM predictions ORDER BY id DESC LIMIT {RECENT_ROWS}", conn
        )
    if df.empty:
        raise HTTPException(status_code=404, detail="No prediction rows in DB yet.")
    df["nameOrig"]       = "C000000000"
    df["nameDest"]       = "C000000000"
    df["isFlaggedFraud"] = 0
    X = _pipeline.transform(df)
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    return X[FEATURE_COLUMNS]


def _run_evidently() -> dict[str, Any]:
    """Run Evidently DataDriftPreset and persist HTML; return structured result dict."""
    production = _load_recent_production()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=_reference_df, current_data=production)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str   = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    html_path = REPORTS_DIR / f"drift_{ts_str}.html"
    report.save_html(str(html_path))

    result_dict   = report.as_dict()
    dataset_meta  = result_dict["metrics"][0]["result"]
    feature_metas = result_dict["metrics"][1]["result"]["drift_by_columns"]

    feature_scores: dict[str, dict] = {
        feat: {
            "drift_score":    round(info.get("drift_score", 0.0), 6),
            "drift_detected": info.get("drift_detected", False),
        }
        for feat, info in feature_metas.items()
    }
    drifted_columns = [f for f, v in feature_scores.items() if v["drift_detected"]]
    alert_count     = len(drifted_columns)

    return {
        "result": {
            "dataset_drifted":  dataset_meta.get("dataset_drift", False),
            "drifted_columns":  drifted_columns,
            "feature_scores":   feature_scores,
            "run_timestamp":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "html_path":    html_path,
        "alert_count":  alert_count,
        "timestamp":    datetime.datetime.now(),
    }


def _get_cached_or_recompute() -> dict[str, Any]:
    """Return cached drift result if < 1 hr old, else recompute."""
    now = datetime.datetime.now()
    if _cache and (now - _cache["timestamp"]).total_seconds() < CACHE_TTL_SECS:
        return _cache
    new_data = _run_evidently()
    _cache.update(new_data)
    return _cache


@router.get("/drift-report")
def get_drift_report() -> dict:
    """Return a JSON drift summary, recomputing if cache is stale."""
    try:
        cached = _get_cached_or_recompute()
        return cached["result"]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift computation failed: {exc}") from exc


@router.get("/drift-report/html")
def get_drift_report_html() -> FileResponse:
    """Stream the latest Evidently HTML drift report as a file download."""
    try:
        cached = _get_cached_or_recompute()
        html_path: Path = cached["html_path"]
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="HTML report file not found.")
        return FileResponse(
            path=str(html_path),
            media_type="text/html",
            filename=html_path.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not serve HTML report: {exc}") from exc


@router.get("/drift-report/status")
def get_drift_status() -> dict:
    """Return lightweight status: last run time, alert count, and severity level."""
    if not _cache:
        return {"last_run": None, "alert_count": 0, "status": "never_run"}
    alert_count: int = _cache["alert_count"]
    if alert_count == 0:
        status = "ok"
    elif alert_count <= 3:
        status = "warning"
    else:
        status = "critical"
    return {
        "last_run":    _cache["result"]["run_timestamp"],
        "alert_count": alert_count,
        "status":      status,
    }
