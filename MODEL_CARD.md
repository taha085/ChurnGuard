# Model Card — ChurnGuard

## Intended use
Prioritize human review of existing telecom customers who may be at risk of churn. Predictions are decision support, not automated cancellation or adverse action.

## Data
IBM Telco Customer Churn public sample. Target `Churn`; positive class is churn. Features include tenure, contract, payment method, service subscriptions, demographics and charges. `customerID` is excluded.

## Method
Feature engineering is documented in `src/features.py`. Imputation, scaling and one-hot encoding are fitted inside a training-only sklearn pipeline. Model selection compares majority baseline, logistic regression, random forest and XGBoost with stratified 5-fold CV. XGBoost is tuned with RandomizedSearchCV.

## Metrics
Populate from `artifacts/evaluation.json` after running `python src/train.py`. Report full classification report, confusion matrix, ROC-AUC, PR-AUC, Brier score and calibration buckets. Do not invent results.

## Limitations
Cross-sectional dataset; random split does not establish future-month performance. Historical dataset may not represent current customers. Risk probabilities require calibration checks. Retention actions and their outcomes are not included.

## Non-use
Do not use as sole basis for denying service, changing prices, or making eligibility decisions. Human review is required.
