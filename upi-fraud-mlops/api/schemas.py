from pydantic import BaseModel
from typing import List, Dict, Any


class TransactionInput(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    risk_band: str
    top_3_reasons: List[Dict[str, Any]]
    model_version: str
    scored_at: str
