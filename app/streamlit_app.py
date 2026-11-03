
import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)

API_URL = os.getenv(
    "CHURNGUARD_API_URL",
    "http://127.0.0.1:8000"
)

API_KEY = os.getenv("CHURNGUARD_API_KEY", "")

st.title("🛡️ ChurnGuard")
st.subheader("Customer Churn Prediction Dashboard")

st.write(
    "Enter customer details below to estimate their likelihood of leaving."
)

with st.form("customer_form"):

    st.header("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
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
        response = requests.post(
            f"{API_URL}/predict",
            json=customer,
            headers={"X-API-Key": API_KEY},
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            st.divider()
            st.header("Prediction Results")

            probability = result["churn_probability"]
            risk = result["risk_label"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Risk Classification",
                    risk.upper()
                )

            st.progress(min(max(probability, 0.0), 1.0))

            st.subheader("Top Risk Drivers")

            drivers = result.get("top_drivers", [])

            if drivers:
                driver_df = pd.DataFrame(drivers)
                st.dataframe(
                    driver_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No driver explanations returned.")

            st.caption(
                "This prediction is a decision-support estimate, "
                "not a guarantee that a customer will churn."
            )

        else:
            st.error(
                f"API Error {response.status_code}: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(
            "Could not connect to FastAPI. Make sure the API server is running."
        )
        st.code(str(e))
