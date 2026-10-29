
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to estimate churn risk. "
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Check models/churn_model.joblib in GitHub.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# Get the input feature names saved by the trained model's pipeline.
if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
elif hasattr(model, "named_steps"):
    last_step = list(model.named_steps.values())[-1]
    if hasattr(last_step, "feature_names_in_"):
        features = list(last_step.feature_names_in_)
    else:
        st.error(
            "The model does not expose its input feature names. "
            "We need to check how it was trained."
        )
        st.stop()
else:
    st.error(
        "The model does not expose its input feature names. "
        "We need to check how it was trained."
    )
    st.stop()

# Typical input choices for the Telco Customer Churn dataset.
categorical_options = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

numeric_defaults = {
    "SeniorCitizen": (0, 0, 1),
    "tenure": (12, 0, 72),
    "MonthlyCharges": (70.0, 0.0, 200.0),
    "TotalCharges": (800.0, 0.0, 10000.0)
}

st.subheader("Customer information")
values = {}
cols = st.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        if feature in numeric_defaults:
            default, minimum, maximum = numeric_defaults[feature]
            if feature in ("SeniorCitizen", "tenure"):
                values[feature] = st.number_input(
                    feature,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1
                )
            else:
                values[feature] = st.number_input(
                    feature,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default)
                )
        elif feature in categorical_options:
            values[feature] = st.selectbox(
                feature,
                categorical_options[feature],
                key=feature
            )
        else:
            st.warning(
                f"Input field '{feature}' needs its valid options configured."
            )
            values[feature] = st.text_input(feature, key=feature)

if st.button("Predict churn risk", type="primary"):
    input_df = pd.DataFrame([values], columns=features)

    try:
        prediction = int(model.predict(input_df)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        if prediction == 1:
            st.error("Prediction: Customer is at risk of churn.")
        else:
            st.success("Prediction: Customer is predicted to stay.")

        if probability is not None:
            st.metric(
                "Estimated churn probability",
                f"{probability:.1%}"
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "The form inputs must match the columns and data types "
            "used when the model was trained."
        )

st.caption(
    "Demonstration only. Validate model quality and consider customer "
    "context before taking action."
)
