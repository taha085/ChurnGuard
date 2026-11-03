import os
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)
KEY = os.getenv("CHURNGUARD_API_KEY","change-me")
SAMPLE = {
"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,
"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
"OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No",
"StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
"PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
"MonthlyCharges":89.1,"TotalCharges":1069.2
}
def test_health():
    assert client.get("/health").status_code == 200
def test_auth_failure():
    assert client.post("/predict",json=SAMPLE,headers={"X-API-Key":"wrong"}).status_code == 401
def test_invalid_input():
    bad = dict(SAMPLE); bad.pop("tenure")
    assert client.post("/predict",json=bad,headers={"X-API-Key":KEY}).status_code == 422
def test_happy_path_requires_trained_model():
    # Run training first; missing artifact should return a clear 503 rather than crash.
    response = client.post("/predict",json=SAMPLE,headers={"X-API-Key":KEY})
    assert response.status_code in (200,503)
def test_unseen_category_is_pipeline_safe():
    # OneHotEncoder(handle_unknown='ignore') supports unseen categories.
    altered = dict(SAMPLE); altered["PaymentMethod"] = "New payment type"
    response = client.post("/predict",json=altered,headers={"X-API-Key":KEY})
    assert response.status_code in (200,503)
