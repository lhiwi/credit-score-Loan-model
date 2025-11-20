from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .data_processing import build_features_pipeline, DATA_DIR


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_features(filename: str = "customer_rfm.csv") -> pd.DataFrame:
    processed_path = DATA_DIR / "processed" / filename
    return pd.read_csv(processed_path)


def train_baseline_model():
    """
    Train a very simple baseline logistic regression model on RFM features.

    For now, we use a dummy target (all zeros) just to make CI/tests pass.
    Later we will replace this with the real proxy target (is_high_risk).
    """
    # ensure features exist
    build_features_pipeline()
    df = load_features()

    # TEMP: dummy target to keep pipeline valid
    # later we'll replace with 'is_high_risk' once we engineer it
    df["dummy_target"] = 0

    X = df[["Recency", "Frequency", "Monetary"]]
    y = df["dummy_target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(model, model_path)

    return model_path


if __name__ == "__main__":
    train_baseline_model()
