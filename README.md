# Credit Risk Scoring and Loan Decision Model

**An End-to-End Machine Learning and Deployment Project**

##  Project Overview

This project develops an **end-to-end credit risk scoring system** using alternative transactional data from an e-commerce platform.
The goal is to assess customer creditworthiness for a **Buy-Now-Pay-Later (BNPL)** service by estimating the probability that a customer is **high-risk** or **low-risk**.

The solution follows a full data science and MLOps lifecycle:

* Business understanding grounded in **credit risk principles**
* Exploratory Data Analysis (EDA)
* Feature engineering using **RFM (Recency, Frequency, Monetary) analysis**
* Proxy target variable construction (since no default label exists)
* Model training and comparison
* Model explainability using **SHAP**
* API deployment using **FastAPI**
* Containerization using **Docker**
* Continuous Integration with **GitHub Actions**

---

##  Business Context: Credit Scoring

Traditional credit scoring relies on historical loan repayment data.
However, in many emerging or digital financial contexts, such labels are unavailable.

In this project:

* **Customer transaction behavior** is used as a proxy for credit risk.
* Customers with **low engagement** (low frequency, low monetary value, long inactivity) are treated as **high-risk**.
* This aligns with principles from the **Basel II Capital Accord**, which emphasizes:

  * Risk measurement
  * Model transparency
  * Documentation and interpretability

---

## Dataset

**Source:** Xente Challenge (Kaggle)

Key data characteristics:

* Transaction-level data
* Customer identifiers
* Monetary values and timestamps
* Product, channel, and pricing information

Files used:

* `training.csv` – main dataset for modeling
* Other CSVs used for reference and validation

---

## Exploratory Data Analysis (EDA)

EDA was conducted in a Jupyter notebook to:

* Understand data structure and distributions
* Identify skewness and outliers in transaction amounts
* Analyze categorical feature dominance
* Explore temporal transaction patterns
* Assess correlations among numerical variables

Key findings:

* Transaction values are **highly right-skewed**
* A small subset of customers accounts for a large share of transactions
* Monetary and frequency patterns vary significantly across customers
* Raw features require aggregation and transformation before modeling

---

##  Feature Engineering

Feature engineering is based on **RFM analysis**, a well-established behavioral modeling approach.

For each customer:

* **Recency** – Days since last transaction
* **Frequency** – Total number of transactions
* **Monetary** – Total transaction value

Transformations applied:

* Log transformation for skewed features
* Scaling to ensure fair clustering and modeling
* Aggregation from transaction-level to customer-level

---

##  Proxy Target Variable Construction

Since no explicit default label exists:

1. Customers are clustered using **K-Means** on RFM features.
2. Clusters are analyzed based on engagement:

   * Low frequency
   * Low monetary value
   * High recency
3. The least engaged cluster is labeled as **high-risk**.

This results in a binary target variable:

```text
is_high_risk = 1  → High risk
is_high_risk = 0  → Low risk
```

---

## Model Training and Evaluation

Two models were trained and compared:

### 1. Logistic Regression (Baseline)

* Interpretable
* Transparent coefficients
* Strong regulatory alignment

### 2. Random Forest (Final Model)

* Captures non-linear relationships
* Higher predictive performance
* Robust to feature interactions

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

**Result:**
The Random Forest model outperformed Logistic Regression while maintaining reasonable interpretability through SHAP.

---

##  Model Explainability (SHAP)

Explainability is critical in financial decision-making.

Using **SHAP (SHapley Additive exPlanations)**:

* **Global explanations** identify which features most influence risk predictions
* **Local explanations** show why a specific customer is classified as high-risk or low-risk

Key insights:

* **Recency** is the strongest risk indicator
* Low transaction frequency increases predicted risk
* Higher monetary engagement reduces risk
* Behavioral patterns dominate over static attributes

These explanations enable **transparent loan decisions** and support regulatory compliance.

---

##  API Deployment (FastAPI)

The trained model is served via a REST API built with **FastAPI**.

### Available Endpoints

| Endpoint   | Description              |
| ---------- | ------------------------ |
| `/`        | API welcome message      |
| `/health`  | Health check             |
| `/predict` | Predict risk probability |
| `/docs`    | Interactive Swagger UI   |

### Example Prediction Response

```json
{
  "risk_probability": 0.94,
  "risk_label": "High Risk"
}
```

---

## 🐳 Containerization (Docker)

The entire application is containerized using Docker.

Included files:

* `Dockerfile`
* `docker-compose.yml`

### Run with Docker

```bash
docker compose up --build
```

Then access:

* [http://127.0.0.1:8000](http://127.0.0.1:8000)
* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Continuous Integration (CI)

GitHub Actions is configured to run on every push:

* Code linting with **flake8**
* Unit tests with **pytest**

This ensures:

* Code quality
* Reproducibility
* Early error detection

---

##  Project Structure

```
credit-score-Loan-model/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── data_processing.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
│
├── tests/
│   └── test_data_processing.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

---

##  Ethical Considerations

* Proxy labels may introduce bias
* Behavioral data must be used responsibly
* Unconventional features (e.g., writing style) should **not** be used without strong ethical and legal justification
* Explainability is essential for customer trust and fair decision-making


---

* Polish it for **submission grading**
* Add **screenshots and diagrams** guidance
