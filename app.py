import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD FILES
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

st.title("💼 Customer Churn Prediction")

# =========================
# USER INPUTS
# =========================
tenure = st.number_input("Tenure", 0, 100, 12)
monthly = st.number_input("Monthly Charges", 0.0, 10000.0, 3000.0)
total = st.number_input("Total Charges", 0.0, 500000.0, 20000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech = st.selectbox("Tech Support", ["Yes", "No"])

# =========================
# CREATE DATAFRAME
# =========================
input_dict = {
    "gender": gender,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "InternetService": internet,
    "TechSupport": tech
}

input_df = pd.DataFrame([input_dict])

# =========================
# MATCH TRAINING COLUMNS
# =========================
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# =========================
# ENCODE
# =========================
for col in encoders:
    if col in input_df.columns:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

# =========================
# SCALE
# =========================
input_scaled = scaler.transform(input_df)

# =========================
# PREDICT
# =========================
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error("⚠️ Customer will churn")
    else:
        st.success("✅ Customer will stay")