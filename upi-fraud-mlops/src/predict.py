"""
predict.py
----------
Single-transaction inference for UPI Fraud Detection.

Loads pipeline.pkl and model.pkl independently (zero dependency on
training code) and:
  1. Accepts a raw transaction as a JSON string or file
  2. Runs it through pipeline.pkl for feature engineering
  3. Scores it with model.pkl
  4. Returns fraud probability
  5. Explains the top 3 SHAP reasons for the decision

Usage:
    python src/predict.py --transaction '{"step":700,"type":"TRANSFER","amount":181.0,
        "nameOrig":"C1305486145","oldbalanceOrg":181.0,"newbalanceOrig":0.0,
        "nameDest":"C553264065","oldbalanceDest":0.0,"newbalanceDest":0.0,
        "isFlaggedFraud":0}'

    # Or from a file:
    python src/predict.py --transaction-file transaction.json
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features import FEATURE_COLUMNS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

THRESHOLD     = 0.3
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
TOP_N_REASONS = 3


# ---------------------------------------------------------------------------
# Artefact loading (done once; in a real service you'd cache at startup)
# ---------------------------------------------------------------------------

def load_artefacts(models_dir: str = MODELS_DIR):
    pipeline = joblib.load(os.path.join(models_dir, "pipeline.pkl"))
    model    = joblib.load(os.path.join(models_dir, "model.pkl"))
    log.info("pipeline.pkl and model.pkl loaded successfully.")
    return pipeline, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(transaction: dict, pipeline, model, top_n: int = TOP_N_REASONS) -> dict:
    """
    Parameters
    ----------
    transaction : dict
        Raw transaction in PaySim schema.  Must contain at minimum:
        step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
        nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud.
        (isFraud is NOT required at inference time.)

    Returns
    -------
    dict with keys:
        fraud_probability : float  (0–1)
        decision          : str    ("FRAUD" | "LEGITIMATE")
        threshold_used    : float
        top_reasons       : list of dicts {"feature", "shap_value", "direction"}
    """

    # ── 1. Build single-row DataFrame ─────────────────────────────────────────
    df_raw = pd.DataFrame([transaction])

    # ── 2. Feature engineering ────────────────────────────────────────────────
    X = pipeline.transform(df_raw)

    # Enforce canonical column order + fill any missing cols
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]

    # ── 3. Fraud probability ──────────────────────────────────────────────────
    fraud_prob = float(model.predict_proba(X)[:, 1][0])
    decision   = "FRAUD" if fraud_prob >= THRESHOLD else "LEGITIMATE"

    # ── 4. SHAP explanations ──────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # fraud class, first (only) row
    else:
        sv = shap_values[0]

    # Sort by absolute SHAP magnitude; return top N
    feature_names = list(X.columns)
    shap_pairs = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    top_reasons = [
        {
            "feature":    feat,
            "shap_value": round(float(val), 6),
            "direction":  "↑ increases fraud risk" if val > 0 else "↓ decreases fraud risk",
            "feat_value": round(float(X[feat].values[0]), 6),
        }
        for feat, val in shap_pairs
    ]

    result = {
        "fraud_probability": round(fraud_prob, 6),
        "decision":          decision,
        "threshold_used":    THRESHOLD,
        "top_reasons":       top_reasons,
    }
    return result


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_result(result: dict):
    prob = result["fraud_probability"]
    dec  = result["decision"]
    thr  = result["threshold_used"]

    bar = "=" * 55
    status_icon = "🚨" if dec == "FRAUD" else "✅"

    print(f"\n{bar}")
    print(f"  {status_icon}  Decision         : {dec}")
    print(f"  📊  Fraud Probability : {prob:.4f}  (threshold: {thr})")
    print(f"{bar}")
    print(f"\n  Top {len(result['top_reasons'])} SHAP Reasons:")
    print(f"  {'Feature':<25}  {'SHAP':>9}  {'Feat Value':>12}  Direction")
    print(f"  {'-'*25}  {'-'*9}  {'-'*12}  {'-'*30}")
    for r in result["top_reasons"]:
        print(
            f"  {r['feature']:<25}  {r['shap_value']:>+9.4f}  "
            f"{r['feat_value']:>12.4f}  {r['direction']}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-transaction fraud inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--transaction",
        type=str,
        help="Raw transaction as a JSON string",
    )
    group.add_argument(
        "--transaction-file",
        type=str,
        help="Path to a JSON file containing the transaction",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=MODELS_DIR,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Decision threshold (default: {THRESHOLD})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_REASONS,
        help=f"Number of SHAP reasons to show (default: {TOP_N_REASONS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Parse input ───────────────────────────────────────────────────────────
    if args.transaction:
        transaction = json.loads(args.transaction)
    else:
        with open(args.transaction_file, "r") as f:
            transaction = json.load(f)

    log.info(f"Transaction type: {transaction.get('type', 'UNKNOWN')}, "
             f"amount: {transaction.get('amount', '?')}")

    # ── Load models ───────────────────────────────────────────────────────────
    global THRESHOLD
    THRESHOLD = args.threshold
    pipeline, model = load_artefacts(args.models_dir)

    # ── Predict ───────────────────────────────────────────────────────────────
    result = predict(transaction, pipeline, model, top_n=args.top_n)

    # ── Output ────────────────────────────────────────────────────────────────
    print_result(result)

    # Also print raw JSON for programmatic consumers
    print("Raw JSON output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
