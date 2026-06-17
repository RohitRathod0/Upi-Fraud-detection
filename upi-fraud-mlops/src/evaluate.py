"""
evaluate.py
-----------
Full metrics suite for the UPI Fraud Detection model.

Loads pipeline.pkl and model.pkl independently (as they would be in
production), runs inference on the held-out test split, and prints:

  • PR-AUC
  • KS Statistic
  • Confusion matrix at 0.3 threshold
  • Full classification report
  • Top 10 SHAP feature importances (beeswarm summary)

Usage:
    python src/evaluate.py --data data/PS_20174392719_1491204439457_log.csv
"""

import argparse
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")          # headless – save to file, never show a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features import FEATURE_COLUMNS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

THRESHOLD = 0.3
TRAIN_CUTOFF = 600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_artefacts(models_dir: str):
    pipeline_path = os.path.join(models_dir, "pipeline.pkl")
    model_path    = os.path.join(models_dir, "model.pkl")

    log.info(f"Loading pipeline from: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)

    log.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    return pipeline, model


def prepare_test_set(csv_path: str, pipeline):
    df   = pd.read_csv(csv_path)
    test = df[df["step"] > TRAIN_CUTOFF].copy()
    log.info(f"Test set: {len(test):,} rows  |  "
             f"fraud rate: {test['isFraud'].mean()*100:.4f}%")

    y    = test["isFraud"].values
    X_raw = test.drop(columns=["isFraud"])
    X    = pipeline.transform(X_raw)

    # Enforce canonical column order
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def evaluate_pr_auc(y_true, proba):
    pr_auc = average_precision_score(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
    print_section("PR-AUC  &  ROC-AUC")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    return pr_auc


def evaluate_ks(y_true, proba):
    fraud_scores     = proba[y_true == 1]
    non_fraud_scores = proba[y_true == 0]
    ks_stat, ks_pval = ks_2samp(fraud_scores, non_fraud_scores)

    print_section("KS Statistic")
    print(f"  KS Statistic : {ks_stat:.4f}")
    print(f"  p-value      : {ks_pval:.2e}")
    print(
        f"\n  Interpretation: {'EXCELLENT' if ks_stat > 0.4 else 'GOOD' if ks_stat > 0.3 else 'FAIR'} "
        f"separation between fraud and non-fraud score distributions."
    )
    return ks_stat


def evaluate_confusion_matrix(y_true, proba, threshold: float = THRESHOLD):
    preds = (proba >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print_section(f"Confusion Matrix  (threshold = {threshold})")
    print(f"\n  {'':20s}  Predicted 0   Predicted 1")
    print(f"  {'Actual 0 (legit)':20s}  {tn:>11,}   {fp:>11,}")
    print(f"  {'Actual 1 (fraud)':20s}  {fn:>11,}   {tp:>11,}")
    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"\n  False Positive Rate : {fp/(fp+tn):.4f}  ({fp:,} false alarms)")
    print(f"  Fraud Caught        : {recall*100:.2f}%  ({tp:,} of {tp+fn:,})")

    print_section("Full Classification Report")
    print(classification_report(y_true, preds, digits=4,
                                target_names=["Legit", "Fraud"]))
    return cm


def plot_pr_curve(y_true, proba, out_dir: str):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, color="#e63946",
            label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(y_true.mean(), linestyle="--", color="#457b9d",
               label=f"Baseline ({y_true.mean():.4f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = os.path.join(out_dir, "pr_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"PR curve saved → {out_path}")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(model, X_test: pd.DataFrame, out_dir: str, top_n: int = 10):
    print_section(f"Top {top_n} SHAP Feature Importances")

    # Use a sample for speed (SHAP TreeExplainer is fast, but 6M rows is a lot)
    sample_size = min(5_000, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification LightGBM: shap_values is array or list[2]
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class-1 (fraud) SHAP values
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": X_test.columns, "mean_|shap|": mean_abs_shap})
        .sort_values("mean_|shap|", ascending=False)
        .head(top_n)
    )

    print(importance_df.to_string(index=False))

    # ── Beeswarm plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        sv,
        X_sample,
        max_display=top_n,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "shap_beeswarm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"SHAP beeswarm plot saved → {out_path}")

    return importance_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate UPI Fraud Detection model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data",
                             "PS_20174392719_1491204439457_log.csv"),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models"),
        help="Directory to save plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load artefacts ────────────────────────────────────────────────────────
    pipeline, model = load_artefacts(args.models_dir)

    # ── Prepare test data ─────────────────────────────────────────────────────
    X_test, y_test = prepare_test_set(args.data, pipeline)

    # ── Predict ───────────────────────────────────────────────────────────────
    log.info("Running inference …")
    proba = model.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    evaluate_pr_auc(y_test, proba)
    evaluate_ks(y_test, proba)
    evaluate_confusion_matrix(y_test, proba, threshold=THRESHOLD)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    plot_pr_curve(y_test, proba, args.out_dir)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    compute_shap(model, X_test, args.out_dir, top_n=10)

    print("\n✅  Evaluation complete.")


if __name__ == "__main__":
    main()
