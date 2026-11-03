# Architecture

1. Data ingestion: public Telco CSV -> `data/`.
2. Feature engineering: deterministic, target-independent transformations in `src/features.py`.
3. Training: preprocessing pipeline, stratified 5-fold CV, baseline/model comparison, randomized XGBoost tuning, untouched holdout evaluation.
4. Artifacts: joblib + skops model serialization, JSON evaluation, CSV comparison, local MLflow run.
5. Serving: FastAPI `/predict` and `/health`, API-key header, structured prediction logs.
6. UI: Streamlit collects customer fields and calls backend over HTTP only.
7. Runtime: Docker image; secrets injected through environment variables.
