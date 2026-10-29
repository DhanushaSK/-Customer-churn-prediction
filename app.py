from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "customer_churn_clean.csv"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate churn risk. This is a demonstration model.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Run `python run_project.py` first. It creates the trained model and cleaned dataset.")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)
target = "Churn" if "Churn" in data.columns else "Exited"
features = data.drop(columns=[target])

st.subheader("Customer information")
values = {}
cols = st.columns(2)
for i, col in enumerate(features.columns):
    series = features[col]
    with cols[i % 2]:
        if pd.api.types.is_numeric_dtype(series):
            med = series.median()
            if pd.isna(med):
                med = 0
            values[col] = st.number_input(col, value=float(med))
            if pd.api.types.is_integer_dtype(series):
                values[col] = int(values[col])
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            values[col] = st.selectbox(col, options, key=col)

if st.button("Predict churn risk", type="primary"):
    input_df = pd.DataFrame([values], columns=features.columns)
    prediction = int(model.predict(input_df)[0])
    probability = None
    try:
        probability = float(model.predict_proba(input_df)[0][1])
    except Exception:
        pass
    if prediction == 1:
        st.error("Prediction: Customer is at risk of churn.")
    else:
        st.success("Prediction: Customer is predicted to stay.")
    if probability is not None:
        st.metric("Estimated churn probability", f"{probability:.1%}")
    st.caption("Use this as a decision-support signal. Validate model quality and consider customer context before taking action.")
