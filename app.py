import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("💳 Credit Risk Dashboard")
st.caption("Simple, actionable credit risk analysis")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("credit_data.csv")

        df.replace("NA", np.nan, inplace=True)
        df.fillna(df.median(), inplace=True)
        df["DebtRatio"] = df["DebtRatio"].clip(0, 3)

        features = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "DebtRatio",
            "NumberOfTimes90DaysLate"
        ]

        X = df[features]
        y = df["SeriousDlqin2yrs"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=10,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)
        return model

    except:
        return None

model = load_model()

# =========================
# INPUT SECTION
# =========================
st.markdown("### 📥 Customer Details")

col1, col2 = st.columns(2)

with col1:
    revolving = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
    age = st.number_input("Age", 18, 100, 30)

with col2:
    total_debt = st.number_input("Total Debt", value=20000.0)
    income = st.number_input("Income", value=30000.0)

if income <= 0:
    st.error("Income must be > 0")
    st.stop()

debt_ratio = min(total_debt / income, 3)
late_90 = st.slider("90 Days Late", 0, 10, 0)

st.metric("Debt Ratio", f"{debt_ratio:.2f}")

# =========================
# FALLBACK MODEL
# =========================
def simple_model():
    score = 0
    score += revolving * 0.4
    score += debt_ratio * 0.2
    score += late_90 * 0.1
    score -= age * 0.002
    return max(0, min(score, 1))

# =========================
# PREDICTION
# =========================
def predict(r, a, d, l):
    if model is not None:
        try:
            X = np.array([[r, a, d, l]])
            prob = model.predict_proba(X)[0][1]

            # business rules
            if d > 2.5:
                prob = max(prob, 0.7)
            if l >= 3:
                prob = max(prob, 0.8)

            return prob
        except:
            return simple_model()
    else:
        return simple_model()

# current prediction
prob = predict(revolving, age, debt_ratio, late_90)
risk = prob * 100

# =========================
# WHAT-IF IMPROVEMENT
# =========================
def improved_scenario():
    new_revolving = min(revolving, 0.3)
    new_debt = min(debt_ratio, 0.8)
    new_late = 0

    new_prob = predict(new_revolving, age, new_debt, new_late)
    return new_prob

improved_prob = improved_scenario()
improved_risk = improved_prob * 100

# =========================
# RESULTS
# =========================
st.markdown("---")
st.markdown("### 📊 Risk Analysis")

col3, col4 = st.columns(2)

with col3:
    st.metric("Current Risk", f"{risk:.1f}%")

    if risk > 70:
        st.error("🚨 High Risk")
    elif risk > 40:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk")

with col4:
    st.metric("Improved Risk", f"{improved_risk:.1f}%")

    change = risk - improved_risk

    if change > 0:
        st.success(f"⬇ Risk can reduce by {change:.1f}%")
    else:
        st.info("Already optimized")

# =========================
# SUGGESTIONS
# =========================
st.markdown("### 🧠 Actionable Suggestions")

tips = []

if revolving > 0.3:
    tips.append("Reduce credit utilization below 30%")

if debt_ratio > 1:
    tips.append("Lower debt ratio by reducing loans or increasing income")

if late_90 > 0:
    tips.append("Avoid late payments (use auto-pay)")

if risk > 70:
    tips.append("Avoid new loans until financials improve")

if len(tips) == 0:
    tips.append("Maintain current financial discipline")

for i, tip in enumerate(tips, 1):
    st.write(f"{i}. {tip}")

# =========================
# DOWNLOAD REPORT
# =========================
report = pd.DataFrame({
    "Current Risk (%)": [risk],
    "Improved Risk (%)": [improved_risk],
    "Risk Reduction (%)": [risk - improved_risk],
    "Utilization": [revolving],
    "Debt Ratio": [debt_ratio],
    "Late Payments": [late_90]
})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    "📄 Download Analysis Report",
    csv,
    "risk_analysis.csv",
    "text/csv",
    key="final_download_btn"
)
