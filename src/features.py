"""Leakage-aware feature engineering for the IBM Telco churn snapshot."""
import numpy as np
import pandas as pd

TARGET = "Churn"
DROP_COLUMNS = ["customerID", TARGET]

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    # Convert TotalCharges robustly; blank strings become missing values.
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tenure = pd.to_numeric(df.get("tenure", 0), errors="coerce").fillna(0)
    monthly = pd.to_numeric(df.get("MonthlyCharges", 0), errors="coerce").fillna(0)
    total = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce")
    service_cols = [c for c in [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ] if c in df.columns]
    if service_cols:
        df["total_services"] = sum((df[c].astype(str) == "Yes").astype(int) for c in service_cols)
    df["tenure_bucket"] = pd.cut(
        tenure, bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
    ).astype("object")
    df["charges_per_tenure_month"] = monthly / tenure.replace(0, np.nan)
    df["charges_per_tenure_month"] = df["charges_per_tenure_month"].replace([np.inf, -np.inf], np.nan)
    df["short_tenure_flag"] = (tenure <= 12).astype(int)
    df["month_to_month_flag"] = (df.get("Contract", "Unknown").astype(str) == "Month-to-month").astype(int)
    df["electronic_check_flag"] = (df.get("PaymentMethod", "Unknown").astype(str) == "Electronic check").astype(int)
    # Drop identifiers and target from model inputs; never construct features from Churn.
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
