# UPI Fraud Detection — MLOps Pipeline

Production-grade MLOps system for detecting UPI payment fraud using the PaySim synthetic dataset.

---

## Project Structure

```
upi-fraud-mlops/
├── data/               ← Place PS_20174392719_1491204439457_log.csv here
├── src/
│   ├── features.py     ← Sklearn-compatible feature engineering transformers
│   ├── train.py        ← End-to-end training: split → pipeline → Optuna → LightGBM
│   ├── evaluate.py     ← PR-AUC, KS statistic, confusion matrix, SHAP beeswarm
│   └── predict.py      ← Single-transaction inference with SHAP explanations
├── models/             ← pipeline.pkl and model.pkl saved here after training
├── notebooks/          ← EDA notebooks
├── api/                ← FastAPI serving (Phase 2)
├── monitoring/         ← Evidently drift detection (Phase 3)
├── dashboard/          ← Streamlit dashboard (Phase 4)
├── docker/             ← Dockerfiles (Phase 5)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Place dataset

Download the PaySim CSV and copy it into the `data/` folder:

```
data/PS_20174392719_1491204439457_log.csv
```

### 3 — Train the model

Full training with 50-trial Optuna search (~30–60 min on CPU):

```bash
python src/train.py
```

Quick smoke-test (skips Optuna, uses sensible defaults — ~2 min):

```bash
python src/train.py --skip-tuning
```

Custom paths:

```bash
python src/train.py \
  --data data/PS_20174392719_1491204439457_log.csv \
  --models-dir models/ \
  --n-trials 50
```

### 4 — Evaluate

Loads `models/pipeline.pkl` and `models/model.pkl` independently:

```bash
python src/evaluate.py
```

Outputs:
- PR-AUC and ROC-AUC
- KS statistic
- Confusion matrix at 0.3 threshold
- Top 10 SHAP feature importances
- `models/pr_curve.png`
- `models/shap_beeswarm.png`

### 5 — Single-transaction inference

```bash
python src/predict.py --transaction '{
  "step": 700,
  "type": "TRANSFER",
  "amount": 181.0,
  "nameOrig": "C1305486145",
  "oldbalanceOrg": 181.0,
  "newbalanceOrig": 0.0,
  "nameDest": "C553264065",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0,
  "isFlaggedFraud": 0
}'
```

From a file:

```bash
python src/predict.py --transaction-file transaction.json
```

---

## Feature Engineering

All transforms are sklearn `TransformerMixin` subclasses inside a `Pipeline`,
ensuring identical behaviour at training and inference (no leakage).

| Step | Transformer | Output columns |
|------|-------------|----------------|
| 1 | `BalanceErrorFeatures` | `orig_balance_error`, `dest_balance_error` |
| 2 | `ZeroBalanceFlags` | `is_orig_zero_before`, `is_dest_zero_before` |
| 3 | `LogAmountTransform` | `log_amount` |
| 4 | `TypeEncoder` | `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` |
| 5 | `DropColumns` | removes `nameOrig`, `nameDest`, `isFlaggedFraud`, `step` |

---

## Model Design

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Algorithm | LightGBM | Speed, handles imbalance, native categorical support |
| Imbalance | `scale_pos_weight = neg/pos` | Avoids over-sampling artefacts |
| Tuning | Optuna (50 trials, TPE sampler) | Principled Bayesian search |
| Objective | PR-AUC | Correct metric for heavily imbalanced data (~0.13% fraud) |
| CV strategy | Stratified K-Fold (3-fold) | Preserves class ratio in each fold |
| Train/test split | Time-based: step ≤ 600 / step > 600 | Simulates real deployment; no future leakage |
| Threshold | 0.3 | Lower than 0.5 to improve recall on rare fraud |

---

## Evaluation Metrics

| Metric | Why |
|--------|-----|
| **PR-AUC** | Primary metric; robust to class imbalance |
| **KS Statistic** | Measures separation of fraud/non-fraud score distributions |
| **Confusion Matrix** | Understand FP/FN trade-off at operating threshold |
| **SHAP** | Explains individual predictions; auditable in production |

---

## Roadmap

- [x] Phase 1 — Training pipeline (`src/train.py`, `src/features.py`)
- [x] Phase 1 — Evaluation suite (`src/evaluate.py`, `src/predict.py`)
- [ ] Phase 2 — FastAPI serving (`api/`)
- [ ] Phase 3 — Evidently drift monitoring (`monitoring/`)
- [ ] Phase 4 — Streamlit dashboard (`dashboard/`)
- [ ] Phase 5 — Docker + CI/CD (`docker/`)

---

## Notes

- `pipeline.pkl` and `model.pkl` are saved **separately** by design.
  They can be loaded and versioned independently in production.
- SHAP `TreeExplainer` is used (not KernelExplainer) for performance.
- The predict script samples 5,000 rows for SHAP batch evaluation to keep
  latency reasonable; single-transaction SHAP runs on 1 row and is fast.
