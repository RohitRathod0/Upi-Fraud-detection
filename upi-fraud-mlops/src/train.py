"""
train.py
--------
End-to-end training pipeline for UPI Fraud Detection.

Workflow:
  1. Load raw PaySim CSV
  2. Time-based train/test split (step ≤ 600 → train, step > 600 → test)
  3. Fit the sklearn feature pipeline on TRAIN only (no leakage)
  4. Optuna hyperparameter search (50 trials, optimising PR-AUC on train CV)
  5. Retrain LightGBM on full train set with best params
  6. Persist models/pipeline.pkl and models/model.pkl

Usage:
    python src/train.py --data data/PS_20174392719_1491204439457_log.csv
"""

import argparse
import logging
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Resolve project root so imports work when called from any directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features import build_feature_pipeline, FEATURE_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Silence Optuna's per-trial logs; keep summary only
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    log.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Sanity checks
    required = {
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest",  "oldbalanceDest", "newbalanceDest",
        "isFraud",   "isFlaggedFraud",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fraud_rate = df["isFraud"].mean() * 100
    log.info(f"Class distribution → fraud: {fraud_rate:.4f}%  "
             f"({df['isFraud'].sum():,} fraud / {len(df):,} total)")
    return df


# ---------------------------------------------------------------------------
# 2. Time-based train/test split
# ---------------------------------------------------------------------------

TRAIN_CUTOFF = 600  # steps 1–600 → train; 601+ → test


def time_split(df: pd.DataFrame):
    train = df[df["step"] <= TRAIN_CUTOFF].copy()
    test  = df[df["step"] >  TRAIN_CUTOFF].copy()
    log.info(
        f"Time-based split → train: {len(train):,} rows (steps 1–{TRAIN_CUTOFF})  |  "
        f"test: {len(test):,} rows (steps {TRAIN_CUTOFF + 1}+)"
    )
    log.info(
        f"Train fraud rate: {train['isFraud'].mean()*100:.4f}%  |  "
        f"Test fraud rate:  {test['isFraud'].mean()*100:.4f}%"
    )
    return train, test


# ---------------------------------------------------------------------------
# 3. Feature pipeline fit + transform
# ---------------------------------------------------------------------------

def fit_pipeline(train_df: pd.DataFrame):
    """Fit the feature pipeline on training data only and return it."""
    pipeline = build_feature_pipeline()
    log.info("Fitting feature engineering pipeline on training data …")

    # We pass the full DataFrame (including target) to the pipeline;
    # it only uses the feature columns internally.
    X_train = train_df.drop(columns=["isFraud"])
    pipeline.fit(X_train)
    log.info("Pipeline fitted.")
    return pipeline


def apply_pipeline(pipeline, df: pd.DataFrame):
    X_raw = df.drop(columns=["isFraud"])
    y     = df["isFraud"].values
    X     = pipeline.transform(X_raw)

    # Enforce canonical column ordering for SHAP reproducibility
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]
    return X, y


# ---------------------------------------------------------------------------
# 4. Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos
    log.info(f"scale_pos_weight = {spw:.2f}  (neg={n_neg:,} / pos={n_pos:,})")
    return spw


def run_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_trials: int = 50,
    cv_folds: int = 3,
) -> dict:
    """
    Search over LightGBM hyperparameters, optimising PR-AUC with
    stratified k-fold cross-validation on the training set.
    """
    log.info(f"Starting Optuna search: {n_trials} trials, {cv_folds}-fold CV …")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":        "binary",
            "metric":           "average_precision",
            "verbosity":        -1,
            "boosting_type":    "gbdt",
            "scale_pos_weight": scale_pos_weight,
            "random_state":     42,
            # Tunable hyperparameters
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        fold_scores = []
        for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
            Xf_tr, Xf_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
            yf_tr, yf_val = y_train[fold_train_idx], y_train[fold_val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                Xf_tr, yf_tr,
                eval_set=[(Xf_val, yf_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            proba = model.predict_proba(Xf_val)[:, 1]
            fold_scores.append(average_precision_score(yf_val, proba))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_score = study.best_value
    log.info(f"Best CV PR-AUC: {best_score:.4f}")
    log.info(f"Best hyperparameters: {best}")
    return best


# ---------------------------------------------------------------------------
# 5. Final model training
# ---------------------------------------------------------------------------

def train_final_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    best_params: dict,
    scale_pos_weight: float,
) -> lgb.LGBMClassifier:
    log.info("Training final LightGBM model on full training set …")
    params = {
        "objective":        "binary",
        "metric":           "average_precision",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "random_state":     42,
        **best_params,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    log.info("Final model trained.")
    return model


# ---------------------------------------------------------------------------
# 6. Quick evaluation on held-out test set
# ---------------------------------------------------------------------------

def quick_eval(model, X_test: pd.DataFrame, y_test: np.ndarray):
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        classification_report,
    )
    from scipy.stats import ks_2samp

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)

    fraud_scores    = proba[y_test == 1]
    non_fraud_scores = proba[y_test == 0]
    ks_stat, _ = ks_2samp(fraud_scores, non_fraud_scores)

    preds = (proba >= 0.3).astype(int)
    cm    = confusion_matrix(y_test, preds)

    log.info("=" * 55)
    log.info(f"  PR-AUC (test)       : {pr_auc:.4f}")
    log.info(f"  KS Statistic (test) : {ks_stat:.4f}")
    log.info(f"  Confusion Matrix (threshold=0.3):\n{cm}")
    log.info("=" * 55)
    log.info("\n" + classification_report(y_test, preds, digits=4))


# ---------------------------------------------------------------------------
# 7. Persist artefacts
# ---------------------------------------------------------------------------

def save_artefacts(pipeline, model, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)

    pipeline_path = os.path.join(models_dir, "pipeline.pkl")
    model_path    = os.path.join(models_dir, "model.pkl")

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(model,    model_path)

    log.info(f"Saved pipeline → {pipeline_path}")
    log.info(f"Saved model    → {model_path}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UPI Fraud Detection LightGBM model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data",
                             "PS_20174392719_1491204439457_log.csv"),
        help="Path to the PaySim CSV file",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models"),
        help="Directory to save pipeline.pkl and model.pkl",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna and use sensible defaults (for quick smoke-test)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = load_data(args.data)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    train_df, test_df = time_split(df)

    # ── 3. Fit pipeline ───────────────────────────────────────────────────────
    pipeline = fit_pipeline(train_df)
    X_train, y_train = apply_pipeline(pipeline, train_df)
    X_test,  y_test  = apply_pipeline(pipeline, test_df)

    log.info(f"Feature matrix shape → train: {X_train.shape}, test: {X_test.shape}")
    log.info(f"Feature columns: {list(X_train.columns)}")

    # ── 4. Scale-pos-weight ───────────────────────────────────────────────────
    spw = compute_scale_pos_weight(y_train)

    # ── 5. Hyperparameter tuning ──────────────────────────────────────────────
    if args.skip_tuning:
        log.info("--skip-tuning set. Using default hyperparameters.")
        best_params = {
            "n_estimators":     500,
            "learning_rate":    0.05,
            "num_leaves":       64,
            "max_depth":        8,
            "min_child_samples": 20,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":        0.1,
            "reg_lambda":       0.1,
        }
    else:
        best_params = run_optuna(X_train, y_train, spw, n_trials=args.n_trials)

    # ── 6. Train final model ──────────────────────────────────────────────────
    model = train_final_model(X_train, y_train, best_params, spw)

    # ── 7. Quick eval ─────────────────────────────────────────────────────────
    quick_eval(model, X_test, y_test)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    save_artefacts(pipeline, model, args.models_dir)

    log.info("Training complete. Run `python src/evaluate.py` for full metrics.")


if __name__ == "__main__":
    main()
