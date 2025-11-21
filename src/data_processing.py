import pandas as pd
from pathlib import Path

# Base data directory: <repo_root>/data
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

    This function assumes the input dataframe has at least:
    - 'CustomerId'
    - 'TransactionId'
    - 'TransactionStartTime' (string or datetime)
    - 'Value' (numeric)

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per CustomerId and columns:
        ['CustomerId', 'Recency', 'Frequency', 'Monetary'].
    """
    df = df.copy()

    # Ensure datetime type
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Snapshot date = one day after the last transaction
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

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    filename : str
        Output CSV filename in data/processed.
    """
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / filename
    df.to_csv(processed_path, index=False)


def build_features_pipeline() -> pd.DataFrame:
    """
    End-to-end feature pipeline:
    - load raw data
    - compute RFM features
    - save to data/processed

    Returns
    -------
    pd.DataFrame
        The RFM dataframe.
    """
    raw_df = load_raw_transactions()
    rfm_df = engineer_rfm_features(raw_df)
    save_processed_features(rfm_df)
    return rfm_df


if __name__ == "__main__":
    # Simple manual run for debugging
    try:
        rfm = build_features_pipeline()
        print("RFM features pipeline completed. Shape:", rfm.shape)
    except FileNotFoundError:
        print(
            "Raw file not found. Make sure data/raw/training.csv exists "
            "before running this module directly."
        )
