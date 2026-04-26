import streamlit as st
import numpy as np
import pickle
import shap

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("💳 Credit Default Risk Prediction")
st.write("Predict loan default risk using Machine Learning")

# =========================
# USER INPUTS
# =========================

st.subheader("📥 Enter Customer Details")

revolving = st.slider("Revolving Utilization (0–1)", 0.0, 1.0, 0.5)
age = st.number_input("Age", min_value=18, max_value=100, value=30)

st.subheader("💡 Debt Ratio Calculator")

total_debt = st.number_input("Total Monthly Debt", min_value=0.0, value=20000.0)
income = st.number_input("Monthly Income", min_value=1.0, value=30000.0)

debt = total_debt / income
st.write(f"👉 Calculated Debt Ratio: **{debt:.2f}**")

late_90 = st.slider("90 Days Late (Serious Delays)", 0, 10, 0)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("🚀 Predict Risk"):

    # IMPORTANT: EXACT 10 FEATURES (match training)
    input_data = np.array([[ 
        revolving,
        age,
        0,              # NumberOfTime30-59DaysPastDueNotWorse
        debt,
        income,
        5,              # NumberOfOpenCreditLinesAndLoans
        late_90,
        1,              # NumberRealEstateLoansOrLines
        0,              # NumberOfTime60-89DaysPastDueNotWorse
        1               # NumberOfDependents
    ]], dtype=float)

    # =========================
    # MODEL PREDICTION
    # =========================
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = probability * 100

    st.subheader("📊 Risk Score")

    if risk_percent >= 75:
        st.error(f"🚨 VERY HIGH RISK ({risk_percent:.1f}%)")
    elif risk_percent >= 50:
        st.warning(f"⚠️ HIGH RISK ({risk_percent:.1f}%)")
    elif risk_percent >= 30:
        st.info(f"⚠️ MODERATE RISK ({risk_percent:.1f}%)")
    else:
        st.success(f"✅ LOW RISK ({risk_percent:.1f}%)")

    # =========================
    # SHAP EXPLANATION
    # =========================
    st.subheader("🧠 Why this prediction?")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        feature_names = [
            "Revolving Utilization",
            "Age",
            "30-59 Days Late",
            "Debt Ratio",
            "Monthly Income",
            "Open Credit Lines",
            "90 Days Late",
            "Real Estate Loans",
            "60-89 Days Late",
            "Dependents"
        ]

        impacts = shap_values.values[0]
        sorted_idx = np.argsort(np.abs(impacts))[::-1]

        for i in sorted_idx[:3]:
            if impacts[i] > 0:
                st.write(f"⬆️ **{feature_names[i]}** is increasing risk")
            else:
                st.write(f"⬇️ **{feature_names[i]}** is reducing risk")

    except:
        st.warning("⚠️ Explanation not available for this input")

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.subheader("💡 Recommendation")

    if risk_percent >= 50:
        st.write("👉 Reduce debt and avoid delayed payments")
        st.write("👉 Improve repayment history before applying for loans")
    else:
        st.write("👉 Maintain current financial behavior")
        st.write("👉 Keep debt ratio low and avoid late payments")
