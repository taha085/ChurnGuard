
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, "/content/src")
from features import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChurnGuard")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/content/artifacts/churnguard_logistic_2026-11-03_v1.joblib"
)

API_KEY = os.getenv("CHURNGUARD_API_KEY")

app = FastAPI(
    title="ChurnGuard API",
    description="Customer Churn Prediction and Retention Support",
    version="1.0.0"
)

model = None


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.on_event("startup")
def load_model():
    global model

    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Logistic Regression model loaded")
    else:
        logger.warning("Model artifact not found")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(
    customer: Customer,
    x_api_key: str = Header(default="")
):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        raw = pd.DataFrame([customer.model_dump()])
        X = engineer_features(raw)

        probability = float(model.predict_proba(X)[0, 1])

        # Explain the largest linear-model contributions.
        transformed = model.named_steps["prep"].transform(X)
        classifier = model.named_steps["model"]

        contributions = (
            transformed.toarray()[0] if hasattr(transformed, "toarray")
            else transformed[0]
        ) * classifier.coef_[0]

        names = model.named_steps["prep"].get_feature_names_out()

        top_indices = sorted(
            range(len(contributions)),
            key=lambda i: abs(contributions[i]),
            reverse=True
        )[:3]

        drivers = [
            {
                "feature": str(names[i]),
                "contribution": round(float(contributions[i]), 4),
                "direction": (
                    "increases risk" if contributions[i] > 0
                    else "reduces risk"
                )
            }
            for i in top_indices
        ]

        response = {
            "churn_probability": round(probability, 4),
            "risk_label": (
                "high-risk" if probability >= 0.40
                else "lower-risk"
            ),
            "threshold": 0.40,
            "top_drivers": drivers
        }

        logger.info(json.dumps({
            "event": "prediction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probability": probability,
            "risk_label": response["risk_label"]
        }))

        return response

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
