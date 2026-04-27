import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk App", layout="wide")

st.title("💳 Credit Risk Predictor")
st.write("Simple & Reliable Risk Estimation")

# =========================
# UI LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUTS
# =========================
with col1:
    st.subheader("📥 Customer Details")

    revolving = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
    age = st.number_input("Age", 18, 100, 30)

    total_debt = st.number_input("Total Debt", value=20000.0)
    income = st.number_input("Income", value=30000.0)

    if income <= 0:
        st.error("Income must be > 0")
        st.stop()

    debt_ratio = total_debt / income

    st.write(f"Debt Ratio: {debt_ratio:.2f}")

    late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# SIMPLE MODEL (NO ML LIBS)
# =========================
def simple_model():
    score = 0

    # risk factors
    score += revolving * 0.4
    score += min(debt_ratio, 5) * 0.2
    score += late_90 * 0.1

    # safe factor
    score -= age * 0.002

    # normalize
    score = max(0, min(score, 1))

    return score

# =========================
# OUTPUT
# =========================
with col2:
    prob = simple_model()
    risk = prob * 100

    st.subheader("📊 Risk Score")

    st.progress(prob)

    if risk > 70:
        st.error(f"🚨 HIGH RISK ({risk:.1f}%)")
    elif risk > 40:
        st.warning(f"⚠️ MEDIUM RISK ({risk:.1f}%)")
    else:
        st.success(f"✅ LOW RISK ({risk:.1f}%)")

    # =========================
    # FEATURE IMPACT
    # =========================
    st.subheader("📊 Key Drivers")

    data = {
        "Feature": ["Utilization", "Debt Ratio", "Late Payments", "Age"],
        "Impact": [
            revolving * 0.4,
            min(debt_ratio, 5) * 0.2,
            late_90 * 0.1,
            -age * 0.002
        ]
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.barh(df["Feature"], df["Impact"])
    ax.invert_yaxis()

    st.pyplot(fig)

    # =========================
    # DOWNLOAD REPORT
    # =========================
    report = pd.DataFrame({
        "Revolving": [revolving],
        "Age": [age],
        "Debt": [total_debt],
        "Income": [income],
        "Late Payments": [late_90],
        "Risk (%)": [risk]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📄 Download Report",
        csv,
        "risk_report.csv",
        "text/csv"
    )
