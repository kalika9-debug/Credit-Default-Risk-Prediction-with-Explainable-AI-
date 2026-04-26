import streamlit as st
import numpy as np
from joblib import load

# SAFE LOAD
try:
    model = load("model.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.title("💳 Credit Risk Prediction")

revolving = st.slider("Revolving Utilization", 0.0, 1.0, 0.5)
age = st.number_input("Age", 18, 100, 30)

total_debt = st.number_input("Total Debt", 0.0, 1000000.0, 20000.0)
income = st.number_input("Income", 1.0, 1000000.0, 30000.0)

debt = total_debt / income
st.write(f"Debt Ratio: {debt:.2f}")

late_90 = st.slider("90 Days Late", 0, 10, 0)

if st.button("Predict"):

    input_data = np.array([[ 
        revolving,
        age,
        0,
        debt,
        income,
        5,
        late_90,
        1,
        0,
        1
    ]])

    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Risk: {prob*100:.2f}%")
