"""
app.py
------
Streamlit dashboard for UPI Fraud Detection MLOps system.

Tabs:
    1. Score Transaction  — POST /score, render risk badge + SHAP bar chart
    2. Drift Monitor      — GET /drift/drift-report, render drift table + alerts
    3. MLflow History     — search_runs() from local tracking DB, metric chart
    4. System Info        — GET /health, architecture overview

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

API_BASE_URL   = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_DB_URI  = f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"
EXPERIMENT     = "upi-fraud-detection"

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🛡️", layout="wide")
st.title("UPI Fraud Detection — MLOps Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Score Transaction", "Drift Monitor", "MLflow History", "System Info"]
)

# ── Tab 1: Score Transaction ──────────────────────────────────────────────────
with tab1:
    st.subheader("Score a UPI Transaction")
    col1, col2 = st.columns(2)
    with col1:
        txn_type    = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount      = st.number_input("Amount (INR)", min_value=1.0, value=50000.0, step=1000.0)
        step        = st.number_input("Step", min_value=1, value=200)
    with col2:
        old_orig    = st.number_input("Sender Opening Balance", min_value=0.0, value=100000.0, step=1000.0)
        new_orig    = st.number_input("Sender Closing Balance", min_value=0.0, value=50000.0, step=1000.0)
        old_dest    = st.number_input("Receiver Opening Balance", min_value=0.0, value=0.0, step=1000.0)
        new_dest    = st.number_input("Receiver Closing Balance", min_value=0.0, value=50000.0, step=1000.0)

    if st.button("Run Fraud Score", type="primary"):
        payload = {
            "step": step, "type": txn_type, "amount": amount,
            "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        }
        try:
            resp = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            prob   = data["fraud_probability"]
            band   = data["risk_band"]
            color  = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[band]

            st.markdown(f"### Risk: :{color}[{band}]  —  Fraud Probability: `{prob:.2%}`")
            st.progress(prob)

            reasons = data["top_3_reasons"]
            features = [r["feature"] for r in reasons]
            values   = [r["value"] for r in reasons]
            colors   = ["#e63946" if v > 0 else "#457b9d" for v in values]

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.barh(features, values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Top 3 SHAP Feature Contributions (fraud class)")
            ax.set_xlabel("SHAP value")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as exc:
            st.error(f"API call failed: {exc}")

# ── Tab 2: Drift Monitor ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Data Drift Monitor (Evidently)")
    if st.button("Refresh Drift Report"):
        try:
            resp = requests.get(f"{API_BASE_URL}/drift/drift-report", timeout=60)
            resp.raise_for_status()
            data = resp.json()

            dataset_drifted = data["dataset_drifted"]
            alert_color     = "red" if dataset_drifted else "green"
            alert_label     = "DATASET DRIFT DETECTED" if dataset_drifted else "No Dataset Drift"
            st.markdown(f"### :{alert_color}[{alert_label}]")

            if data["drifted_columns"]:
                st.warning(f"Drifted features: {', '.join(data['drifted_columns'])}")

            rows = [
                {"Feature": feat, "Drift Score": v["drift_score"], "Drifted": v["drift_detected"]}
                for feat, v in data["feature_scores"].items()
            ]
            df = pd.DataFrame(rows).sort_values("Drift Score", ascending=False)
            st.dataframe(
                df,
                use_container_width=True,
            )
            st.caption(f"Last run: {data['run_timestamp']}")

        except Exception as exc:
            st.error(f"Drift report failed: {exc}")

# ── Tab 3: MLflow History ─────────────────────────────────────────────────────
with tab3:
    st.subheader("MLflow Experiment Runs")
    try:
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        runs_df = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
        if runs_df.empty:
            st.info("No MLflow runs found. Run src/log_existing_run.py first.")
        else:
            display_cols = [c for c in runs_df.columns if c.startswith("metrics.") or c in ("run_id", "status", "start_time")]
            st.dataframe(runs_df[display_cols], use_container_width=True)

            metric_col = "metrics.f1_score"
            if metric_col in runs_df.columns:
                chart_df = runs_df[["start_time", metric_col]].dropna().sort_values("start_time")
                st.line_chart(chart_df.set_index("start_time")[metric_col])

    except Exception as exc:
        st.error(f"MLflow query failed: {exc}")

# ── Tab 4: System Info ────────────────────────────────────────────────────────
with tab4:
    st.subheader("System Info")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        health = resp.json()
        st.success(f"API Status: {health['status'].upper()}  |  Model Version: `{health['model_version']}`")
    except Exception as exc:
        st.error(f"API unreachable: {exc}")

    st.markdown("""
    ### Architecture
    ```
    User → Streamlit (8501)
              ↓
         FastAPI (8000)
        /       |       \\
    LightGBM  Evidently  predictions.db
        \\       |
         MLflow (mlflow.db)
    ```
    **Stack:** LightGBM · Isolation Forest · SHAP · Evidently AI · MLflow · FastAPI · Streamlit · Docker
    """)
