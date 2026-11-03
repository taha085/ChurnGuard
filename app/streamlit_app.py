
import sys
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import engineer_features

# Page configuration
st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model
MODEL_PATH = ROOT / "artifacts/churnguard_logistic_2026-11-03_v1.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Dashboard heading
st.title("🛡️ ChurnGuard")
st.subheader("Customer Churn Prediction Dashboard")

st.write(
    "Enter customer details to estimate their likelihood of leaving."
)

# Customer form
with st.form("customer_form"):

    st.header("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", 0, 100, 12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=89.10
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=1069.20
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True
    )

# Prediction
if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        # Feature engineering
        input_df = pd.DataFrame([customer])
        engineered = engineer_features(input_df)

        # Predict probability
        probability = float(model.predict_proba(engineered)[0, 1])

        threshold = 0.40
        risk = "HIGH-RISK" if probability >= threshold else "LOW-RISK"

        # Transform features for coefficient explanations
        prep = model.named_steps["prep"]
        classifier = model.named_steps["model"]

        transformed = prep.transform(engineered)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = prep.get_feature_names_out()
        contributions = transformed[0] * classifier.coef_[0]

        driver_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions
        })

        driver_df = driver_df.reindex(
            driver_df["contribution"].abs().sort_values(
                ascending=False
            ).index
        ).head(3)

        driver_df["direction"] = np.where(
            driver_df["contribution"] > 0,
            "increases risk",
            "reduces risk"
        )

        driver_df["contribution"] = driver_df[
            "contribution"
        ].round(4)

        # Results
        st.divider()
        st.header("Prediction Results")

        c1, c2 = st.columns(2)

        c1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        c2.metric("Risk Classification", risk)

        st.progress(probability)

        if risk == "HIGH-RISK":
            st.warning("This customer has elevated churn risk.")
        else:
            st.success("This customer has relatively lower churn risk.")

        st.subheader("Top Risk Drivers")

        st.dataframe(
            driver_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Risk threshold: 40%. Driver contributions are "
            "Logistic Regression coefficient-based explanations, "
            "not SHAP values. This is a decision-support estimate."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
