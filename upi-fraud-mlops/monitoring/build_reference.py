"""
build_reference.py
------------------
One-time script: builds the reference dataset for Evidently drift detection.

Loads training CSV → runs through pipeline.pkl → stratified 10k sample →
saves to data/reference_data.parquet.

Run once after training, never again unless you retrain the model.

Usage:
    python monitoring/build_reference.py
"""

import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features import FEATURE_COLUMNS  # noqa: E402

TRAIN_CUTOFF   = 600
SAMPLE_SIZE    = 10_000
CSV_PATH       = os.path.join(PROJECT_ROOT, "data", "PS_20174392719_1491204439457_log.csv")
PIPELINE_PATH  = os.path.join(PROJECT_ROOT, "models", "pipeline.pkl")
OUT_PATH       = os.path.join(PROJECT_ROOT, "data", "reference_data.parquet")


def main():
    print("Loading training data …")
    df = pd.read_csv(CSV_PATH)
    train = df[df["step"] <= TRAIN_CUTOFF].copy()
    print(f"Training rows: {len(train):,}  |  fraud rate: {train['isFraud'].mean()*100:.4f}%")

    print("Applying feature pipeline …")
    pipeline = joblib.load(PIPELINE_PATH)
    y = train["isFraud"].values
    X = pipeline.transform(train.drop(columns=["isFraud"]))
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS].copy()
    X["isFraud"] = y

    print(f"Taking stratified {SAMPLE_SIZE:,} row sample …")
    fraud     = X[X["isFraud"] == 1]
    non_fraud = X[X["isFraud"] == 0]
    n_fraud   = int(SAMPLE_SIZE * fraud.shape[0] / len(X))
    n_legit   = SAMPLE_SIZE - n_fraud
    sample = pd.concat([
        fraud.sample(n=min(n_fraud, len(fraud)), random_state=42),
        non_fraud.sample(n=n_legit, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    sample.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}  ({len(sample):,} rows, fraud: {sample['isFraud'].mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
