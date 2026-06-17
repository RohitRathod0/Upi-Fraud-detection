"""
main.py
-------
FastAPI app for UPI Fraud Detection inference.

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sqlite3
from fastapi import FastAPI
from api.schemas import TransactionInput, ScoreResponse
from api import predictor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "predictions.db")

# Initialize SQLite database
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
with sqlite3.connect(DB_PATH) as conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER,
            type TEXT,
            amount REAL,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            oldbalanceDest REAL,
            newbalanceDest REAL,
            fraud_probability REAL,
            risk_band TEXT,
            model_version TEXT,
            scored_at TEXT
        )
    ''')

app = FastAPI(
    title="UPI Fraud Detection API",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": predictor.MODEL_VERSION}


@app.post("/score", response_model=ScoreResponse)
def score(txn: TransactionInput):
    txn_dict = txn.model_dump()
    result = predictor.score(txn_dict)
    
    # Synchronous SQLite logging
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT INTO predictions (
                step, type, amount, oldbalanceOrg, newbalanceOrig, 
                oldbalanceDest, newbalanceDest, fraud_probability, 
                risk_band, model_version, scored_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            txn.step, txn.type, txn.amount, txn.oldbalanceOrg, txn.newbalanceOrig,
            txn.oldbalanceDest, txn.newbalanceDest, result["fraud_probability"],
            result["risk_band"], result["model_version"], result["scored_at"]
        ))
        
    return ScoreResponse(**result)

