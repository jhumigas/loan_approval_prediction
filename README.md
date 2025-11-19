# Loan Approval Prediction Project

## 1. Problem Statement

This project aims to predict whether a loan application will be **approved or rejected** based on an applicant's demographic, financial, and employment information. 

**Who benefits:** Financial institutions can use this model to streamline their loan approval process, reduce manual review time, and make more consistent decisions. Applicants benefit from faster processing times and clearer understanding of approval factors.

**How the model will be used:** The model serves as a decision-support tool that processes loan applications and provides a binary prediction (approved/rejected) along with a confidence score. It can be integrated into existing loan processing systems via a REST API.

**Why this matters:** Automating initial loan screening can help lenders process applications more efficiently while maintaining fair and consistent evaluation criteria. This reduces operational costs and improves customer experience through faster turnaround times.

## 2. Dataset Description

**Source:** [Kaggle - Loan Approval Dataset](https://www.kaggle.com/datasets/anishdevedward/loan-approval-dataset/data)

**Overview:** The dataset contains 2,000 simulated loan applications with the following characteristics:

- **Target Variable:** `loan_approved` (True/False)
- **Features Include:**
  - Financial: Income, existing loans, credit score
  - Employment: Employment status, years employed

**Data Quality:**
- Size: 2,000 records (manageable for quick iteration)
- Missing values: None
- Class balance: 879 (43.95%) positive, 1121 (56.05%) negative

## 3. EDA Summary

Key findings from exploratory data analysis:

- **Income Distribution:** Highly skewed with most applicants earning between 30K-130K, with some high earners up to 1.5M
- **Credit Score Range:** Spans from 300 to 822, with typical range around 300-822
- **Loan Amounts:** Requested amounts range from ~1,000 to 74,000+
- **Employment Tenure:** Varies from 0 to 39 years
- **Geographic Distribution:** 1,882 unique cities represented
- **Approval Rate:** 43.95%

**Feature Engineering:**
- Created `debt_to_income_ratio`: loan_amount / income
- Binned `credit_score` into categories (Poor/Fair/Good/Excellent)
- Encoded categorical variables (`city` using target encoding or frequency encoding due to high cardinality)
- Dropped `name` and `city` column (not predictive)
- [Add any other transformations you performed]

**Visualizations:** See `notebooks/eda.ipynb` for detailed plots and analysis.

## 4. Modeling Approach & Metrics

**Baseline Model:** Logistic Regression
- Chosen for its interpretability and efficiency
- Provides probability scores useful for risk assessment
- Coefficients reveal feature importance for business insights

**Model Training:**
- Train/validation/test split: 60/20/20
- Cross-validation: 5-fold CV on training set
- Hyperparameter tuning: Regularization strength (C parameter)

**Evaluation Metric:** 
- **Primary:** AUC-ROC (measures ability to distinguish between classes)
- **Secondary:** Precision and Recall (important for understanding false positives vs false negatives in loan context)

**Results:**
| Model              | AUC-ROC | Precision | Recall | Accuracy |
|--------------------|---------|-----------|--------|----------|
| Logistic Regression| 0.84    | 0.84      | 0.83   | 0.83     |

**API Implementation:** FastAPI with automatic OpenAPI documentation

## 5. How to Run Locally

### Prerequisites
- Python 3.11+
- uv (recommended) or pip

### Setup
```bash
# Clone the repository
git clone [your-repo-url]
cd loan-approval-prediction

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using uv
uv sync

# Train the model
uv run python train.py

# Run predictions
uv run python predict.py --input sample_data.json
```

### Start the Web Service
```bash
uv run uvicorn serve:app --host 0.0.0.0 --port 9696
```
The API will be available at `http://localhost:9696`

You can also access the interactive API docs at `http://localhost:9696/docs`

## 6. Running with Docker

### Build the Docker Image
```bash
docker build -t loan-approval-api .
```

### Run the Container
```bash
docker run -it --rm -p 9696:9696 loan-approval-api
```

The service will be accessible at:
- API: `http://localhost:9696`
- Interactive docs: `http://localhost:9696/docs`

## 7. API Usage Example

### Health Check
```bash
curl http://localhost:9696/health
```

**Response:**
```json
{"status": "healthy"}
```

### Make a Prediction
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Port Jesseville",
    "income": 62098,
    "credit_score": 689,
    "loan_amount": 19217,
    "years_employed": 29,
    "points": 65
  }'
```

**Response:**
```json
{
  "loan_approved": true,
  "probability": 0.87,
  "model_version": "1.0"
}
```

**Note:** The `name` field is not required for prediction as it's not used in the model.

## 8. Project Structure

```
loan-approval-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned data
│
├── notebooks/
│   ├── eda.ipynb              # Exploratory analysis
│   └── modeling.ipynb         # Model development
│
├── src/
│   ├── train.py               # Model training script
│   ├── predict.py             # Inference script
│   └── serve.py               # FastAPI web service
│
├── models/
│   └── logistic_model.pkl     # Saved model artifact
│
├── tests/
│   └── test_api.py            # Unit tests
│
├── pyproject.toml             # uv project configuration
├── uv.lock                    # Locked dependencies
├── Dockerfile
├── README.md
└── .gitignore
```

## 9. Architecture Diagram

```
┌─────────────┐
│   Raw Data  │
│  (CSV/JSON) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     EDA     │
│  & Cleaning │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Feature   │
│ Engineering │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Logistic  │
│ Regression  │
│   Training  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Model     │
│  Artifact   │
│  (.pkl)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Flask API  │
│   Service   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Prediction │
│   Response  │
└─────────────┘
```

## 10. Known Limitations & Next Steps

**Current Limitations:**
- Model trained on simulated data; real-world performance may vary
- Does not account for macroeconomic factors (interest rates, market conditions)
- Binary classification only (no risk scoring tiers)

**Future Improvements:**
- Experiment with ensemble methods (Random Forest, XGBoost) for potentially better performance
- Add explainability features (SHAP values) to show why loans were approved/rejected
- Implement A/B testing framework for model versions
- Add monitoring for model drift detection
- Create interactive dashboard for business users

## 11. Dependencies

This project uses **uv** for fast, reliable dependency management.

**Key files:**
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions for reproducibility

**Main libraries:**
- fastapi - Modern web framework for building APIs
- uvicorn - ASGI server for FastAPI
- pandas - Data manipulation
- scikit-learn - Machine learning
- numpy - Numerical computing
- pydantic - Data validation

To view all dependencies:
```bash
uv tree
```

## 12. Model Performance Notes

- The model performs best on applications within typical income ranges
- Edge cases (very high/low income) may require additional validation
- Regular retraining recommended as loan approval patterns evolve

## 13. License

[Specify your license here]

## 14. Contact

[Your name and contact information]

---

**Project Submission Details:**
- Commit Hash: [insert hash]
- Date: [insert date]
- Course: [ML Zoomcamp / other]