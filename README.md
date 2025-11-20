# Credit Scoring and Loan Decision Model (Xente Alternative Data)

This project builds a credit risk probability model using alternative e-commerce transaction data,
with the goal of estimating customer risk and informing loan approval decisions.

## Project Structure

- `data/raw/` – raw Xente CSV files (`training.csv`)
- `data/processed/` – engineered customer-level features (RFM, etc.)
- `notebooks/` – exploratory analysis (`EDA.ipynb`)
- `src/` – production code (data processing, training, API)
- `tests/` – unit tests
- `.github/workflows/ci.yml` – CI pipeline (lint + tests)

## How to Run Locally

```bash
python -m venv .venv
.\.venv\Scripts\activate      # on Windows

pip install -r requirements.txt

# run feature pipeline
python -m src.train
