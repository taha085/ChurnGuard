# ChurnGuard — Customer Churn Prediction & Retention Support System

ChurnGuard is a CPU-friendly capstone scaffold for identifying customers at elevated churn risk, returning a probability, and supporting retention-team triage.

## Quick start
Python 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Training downloads the public IBM Telco churn dataset, creates leakage-audited features, performs 5-fold stratified CV, compares a majority baseline / Logistic Regression / Random Forest / XGBoost, tunes XGBoost with RandomizedSearchCV, evaluates on an untouched stratified holdout, and writes metrics and model artifacts.

Set an API key and start backend:
```bash
# Windows PowerShell: $env:CHURNGUARD_API_KEY="your-long-secret"
# macOS/Linux: export CHURNGUARD_API_KEY="your-long-secret"
uvicorn app.api:app --reload
```
API docs: http://localhost:8000/docs

In another terminal:
```bash
streamlit run app/streamlit_app.py
```

## API
`GET /health` reports service/model availability.
`POST /predict` accepts customer attributes and requires `X-API-Key`. It returns churn probability, risk label, threshold and driver text.

## Important status / limitations
This repository is a build scaffold and must be trained and tested in your environment before submission. Do not claim a public deployment until you deploy it and verify the live URL. The API currently returns concise feature-context driver explanations; full SHAP global and per-customer explanations remain an explicit advanced implementation task. Review model calibration and select the operating threshold against retention-team capacity and intervention cost. A random holdout on this cross-sectional dataset does not prove future-time performance.
