import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when running pytest locally
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import engineer_rfm_features  # noqa: E402


def test_engineer_rfm_features_columns():
    """
    The engineered RFM dataframe should contain the expected columns.
    """
    data = {
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionId": ["T1", "T2", "T3"],
        "TransactionStartTime": [
            "2019-01-01 10:00:00",
            "2019-01-02 11:00:00",
            "2019-01-03 12:00:00",
        ],
        "Value": [100.0, 200.0, 50.0],
    }
    df = pd.DataFrame(data)
    rfm = engineer_rfm_features(df)

    expected_cols = {"CustomerId", "Recency", "Frequency", "Monetary"}
    assert expected_cols.issubset(set(rfm.columns))


def test_engineer_rfm_features_aggregation():
    """
    The RFM aggregation should correctly sum Monetary and count Frequency.
    """
    data = {
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionId": ["T1", "T2", "T3"],
        "TransactionStartTime": [
            "2019-01-01 10:00:00",
            "2019-01-02 11:00:00",
            "2019-01-03 12:00:00",
        ],
        "Value": [100.0, 200.0, 50.0],
    }
    df = pd.DataFrame(data)
    rfm = engineer_rfm_features(df)

    c1 = rfm[rfm["CustomerId"] == "C1"].iloc[0]
    assert c1["Frequency"] == 2
    assert c1["Monetary"] == 300.0
