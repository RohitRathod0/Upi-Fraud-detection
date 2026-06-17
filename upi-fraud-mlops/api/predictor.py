"""
predictor.py
------------
Loads pipeline.pkl and model.pkl once at module level.
Exposes a single score() function used by main.py.
"""

import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap

# Resolve project root and make features.py importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features import FEATURE_COLUMNS  # noqa: E402

# ── Load artefacts once ───────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
pipeline = joblib.load(os.path.join(_MODELS_DIR, "pipeline.pkl"))
model    = joblib.load(os.path.join(_MODELS_DIR, "model.pkl"))
explainer = shap.TreeExplainer(model)

MODEL_VERSION = "lgbm-v1"
THRESHOLD     = 0.3


def _risk_band(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= THRESHOLD:
        return "MEDIUM"
    return "LOW"


def score(txn: dict) -> dict:
    # ── Feature engineering ───────────────────────────────────────────────────
    df = pd.DataFrame([txn])
    X  = pipeline.transform(df)
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]

    # ── Fraud probability ─────────────────────────────────────────────────────
    fraud_prob = float(model.predict_proba(X)[:, 1][0])

    # ── SHAP top-3 reasons ────────────────────────────────────────────────────
    sv = explainer.shap_values(X)
    sv = sv[1][0] if isinstance(sv, list) else sv[0]

    top3 = sorted(
        zip(FEATURE_COLUMNS, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    return {
        "fraud_probability": round(fraud_prob, 6),
        "risk_band":         _risk_band(fraud_prob),
        "top_3_reasons":     [{"feature": f, "value": round(float(v), 6)} for f, v in top3],
        "model_version":     MODEL_VERSION,
        "scored_at":         datetime.now(timezone.utc).isoformat(),
    }
