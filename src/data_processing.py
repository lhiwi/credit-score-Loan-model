import pandas as pd
from pathlib import Path
from typing import Tuple

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw_transactions(filename: str = "training.csv") -> pd.DataFrame:
    """
    Load raw transaction data from data/raw.

    Parameters
    ----------
    filename : str
        Name of the CSV file in data/raw.

    Returns
    -------
    pd.DataFrame
        Raw transactions dataframe.
    """
    raw_path = DATA_DIR / "raw" / filename
    return pd.read_csv(raw_path)


def engineer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic RFM features per CustomerId (Recency, Frequency, Monetary).

    This is a first version; later we'll add clustering and proxy target.
    """
    # ensure datetime
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime",
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum"),
        )
        .reset_index()
    )

    return rfm


def save_processed_features(df: pd.DataFrame,
                            filename: str = "customer_rfm.csv") -> None:
    """
    Save processed customer-level features to data/processed.
    """
    processed_path = DATA_DIR / "processed" / filename
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)


def build_features_pipeline() -> pd.DataFrame:
    """
    End-to-end feature pipeline:
    load raw data -> compute RFM -> save -> return dataframe.
    """
    raw_df = load_raw_transactions()
    rfm_df = engineer_rfm_features(raw_df)
    save_processed_features(rfm_df)
    return rfm_df
