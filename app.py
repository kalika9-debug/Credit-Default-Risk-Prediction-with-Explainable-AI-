import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk App", layout="wide")

st.title("💳 Credit Risk Predictor")
st.write("Now powered with ML + fallback safety")

# =========================
# LOAD / TRAIN MODEL SAFELY
# =========================
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("credit_data.csv")

        features = [
            "RevolvingUtilization",
            "Age",
            "DebtRatio",
            "Late90"
        ]

        X = df[features]
        y = df["Default"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(X_scaled, y)

        return model, scaler

    except Exception as e:
        return None, None

model, scaler = load_model()

# =========================
# UI
# =========================
col1, col2 = st.columns(2)

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
# FALLBACK MODEL
# =========================
def simple_model():
    score = 0
    score += revolving * 0.4
    score += min(debt_ratio, 5) * 0.2
    score += late_90 * 0.1
    score -= age * 0.002
    return max(0, min(score, 1))

# =========================
# PREDICTION LOGIC
# =========================
def predict():
    if model is not None:
        try:
            X = np.array([[revolving, age, debt_ratio, late_90]])
            X_scaled = scaler.transform(X)
            prob = model.predict_proba(X_scaled)[0][1]
            return prob, "ML Model"
        except:
            return simple_model(), "Fallback Model"
    else:
        return simple_model(), "Fallback Model"

# =========================
# OUTPUT
# =========================
with col2:
    prob, mode = predict()
    risk = prob * 100

    st.subheader("📊 Risk Score")
    st.caption(f"Mode: {mode}")

    st.progress(prob)

    if risk > 70:
        st.error(f"🚨 HIGH RISK ({risk:.1f}%)")
    elif risk > 40:
        st.warning(f"⚠️ MEDIUM RISK ({risk:.1f}%)")
    else:
        st.success(f"✅ LOW RISK ({risk:.1f}%)")

    # =========================
    # FEATURE IMPACT (SAFE)
    # =========================
    st.subheader("📊 Key Drivers")

    impacts = [
        revolving * 0.4,
        min(debt_ratio, 5) * 0.2,
        late_90 * 0.1,
        -age * 0.002
    ]

    df = pd.DataFrame({
        "Feature": ["Utilization", "Debt Ratio", "Late Payments", "Age"],
        "Impact": impacts
    })

    fig, ax = plt.subplots()
    ax.barh(df["Feature"], df["Impact"])
    ax.invert_yaxis()

    st.pyplot(fig)

    # =========================
    # DOWNLOAD
    # =========================
    report = pd.DataFrame({
        "Revolving": [revolving],
        "Age": [age],
        "Debt": [total_debt],
        "Income": [income],
        "Late Payments": [late_90],
        "Risk (%)": [risk],
        "Model Used": [mode]
    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📄 Download Report",
        csv,
        "risk_report.csv",
        "text/csv"
    )
