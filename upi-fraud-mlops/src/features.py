"""
features.py
-----------
All feature engineering for the UPI Fraud Detection pipeline.

Design principles:
  - Every transform is an sklearn BaseEstimator + TransformerMixin so it
    fits inside a Pipeline and runs identically at training AND inference.
  - No data leakage: all features derive only from columns present at the
    moment a transaction is INITIATED (no future look-ahead).
  - The full column order is deterministic, which is critical for SHAP.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Step 1 – Balance-error features
# ---------------------------------------------------------------------------

class BalanceErrorFeatures(BaseEstimator, TransformerMixin):
    """
    Derives two balance-consistency checks that are hallmarks of fraud:

    orig_balance_error  = oldbalanceOrg - newbalanceOrig - amount
        Ideally 0 for a legitimate transfer. Non-zero means the
        originator's account balance does not add up correctly.

    dest_balance_error  = oldbalanceDest + amount - newbalanceDest
        Ideally 0 for a legitimate transfer into the destination.
        Non-zero signals manipulation on the receiving side.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["orig_balance_error"] = (
            X["oldbalanceOrg"] - X["newbalanceOrig"] - X["amount"]
        )
        X["dest_balance_error"] = (
            X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
        )
        return X


# ---------------------------------------------------------------------------
# Step 2 – Zero-balance flags
# ---------------------------------------------------------------------------

class ZeroBalanceFlags(BaseEstimator, TransformerMixin):
    """
    Boolean flags for accounts that start a transaction with zero balance.
    These strongly correlate with mule/burner accounts used in fraud.

    is_orig_zero_before : 1 if oldbalanceOrg == 0 else 0
    is_dest_zero_before : 1 if oldbalanceDest == 0 else 0
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["is_orig_zero_before"] = (X["oldbalanceOrg"] == 0).astype(int)
        X["is_dest_zero_before"] = (X["oldbalanceDest"] == 0).astype(int)
        return X


# ---------------------------------------------------------------------------
# Step 3 – Log-transform of amount
# ---------------------------------------------------------------------------

class LogAmountTransform(BaseEstimator, TransformerMixin):
    """
    Applies log1p to `amount` to compress the extreme right-skew.
    The raw `amount` column is preserved (renamed) for completeness;
    the model will primarily use log_amount.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["log_amount"] = np.log1p(X["amount"])
        return X


# ---------------------------------------------------------------------------
# Step 4 – Drop columns that are not model features
# ---------------------------------------------------------------------------

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Removes columns that must NOT be seen by the model:
      - nameOrig, nameDest  : high-cardinality IDs, not generalisable
      - isFlaggedFraud      : rule-based flag that leaks the label space
      - step                : time index used only for train/test split
    """

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or [
            "nameOrig", "nameDest", "isFlaggedFraud", "step"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols)


# ---------------------------------------------------------------------------
# Step 5 – One-hot encode transaction type
# ---------------------------------------------------------------------------

class TypeEncoder(BaseEstimator, TransformerMixin):
    """
    One-hot encodes the `type` column.

    PaySim transaction types:  CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    Only TRANSFER and CASH_OUT carry fraud, so we keep all dummies but
    drop the first (CASH_IN) to avoid the dummy-variable trap.

    Output columns (after drop_first):
        type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
    """

    KNOWN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dummies = pd.get_dummies(
            X["type"],
            prefix="type",
            drop_first=True,          # drops type_CASH_IN
        )
        # Guarantee all expected columns are present (handles unseen types)
        for col in [f"type_{t}" for t in self.KNOWN_TYPES[1:]]:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies.astype(int)
        X = pd.concat([X.drop(columns=["type"]), dummies], axis=1)
        return X


# ---------------------------------------------------------------------------
# Assembled feature pipeline (returns a pd.DataFrame)
# ---------------------------------------------------------------------------

def build_feature_pipeline() -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline that sequentially applies:
      1. BalanceErrorFeatures
      2. ZeroBalanceFlags
      3. LogAmountTransform
      4. TypeEncoder
      5. DropColumns

    The pipeline input must be a pandas DataFrame with the raw PaySim
    columns.  The output is a pandas DataFrame ready for the LightGBM
    model (all numeric, no string columns).
    """
    return Pipeline(
        steps=[
            ("balance_errors",   BalanceErrorFeatures()),
            ("zero_flags",       ZeroBalanceFlags()),
            ("log_amount",       LogAmountTransform()),
            ("type_encoder",     TypeEncoder()),
            ("drop_cols",        DropColumns()),
        ]
    )


# ---------------------------------------------------------------------------
# Helper – canonical feature name list (for SHAP column alignment)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "is_orig_zero_before",
    "is_dest_zero_before",
    "log_amount",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]
