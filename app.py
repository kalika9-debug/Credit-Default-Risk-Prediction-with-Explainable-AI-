import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =========================
# PAGE CONFIG + STYLE
# =========================
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.section-box {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
h2 {color: #1f3b4d;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>💳 Credit Risk Dashboard</h2>", unsafe_allow_html=True)
st.caption("AI-powered credit risk analysis with actionable insights")

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
            n_estimators=150,
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
def predict():
    if model is not None:
        try:
            X = np.array([[revolving, age, debt_ratio, late_90]])
            prob = model.predict_proba(X)[0][1]

            # business rules
            if debt_ratio > 2.5:
                prob = max(prob, 0.7)
            if late_90 >= 3:
                prob = max(prob, 0.8)

            return prob, "XGBoost"
        except:
            return simple_model(), "Fallback"
    else:
        return simple_model(), "Fallback"

prob, mode = predict()
risk = prob * 100

# =========================
# RESULT SECTION
# =========================
st.markdown("---")
st.markdown("### 📊 Risk Analysis")

col3, col4 = st.columns([1,2])

with col3:
    st.metric("Default Risk", f"{risk:.1f}%")
    st.caption(f"Model: {mode}")

    if risk > 70:
        st.error("🚨 High Risk")
    elif risk > 40:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk")

with col4:
    st.progress(prob)

# =========================
# FEATURE IMPORTANCE
# =========================
if model is not None:
    st.markdown("### 📊 Key Drivers")

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": ["Utilization", "Age", "Debt Ratio", "Late Payments"],
        "Importance": importance
    }).sort_values(by="Importance")

    st.bar_chart(df_imp.set_index("Feature"))

# =========================
# RECOMMENDATIONS
# =========================
st.markdown("### 🧠 Personalized Suggestions")

tips = []

# Utilization
if revolving > 0.8:
    tips.append("🚨 Reduce credit usage immediately.")
elif revolving > 0.5:
    tips.append("⚠️ Keep utilization below 30%.")

# Debt
if debt_ratio > 2.5:
    tips.append("🚨 Reduce debt aggressively or increase income.")
elif debt_ratio > 1:
    tips.append("⚠️ Avoid taking new loans.")

# Late payments
if late_90 >= 3:
    tips.append("🚨 Multiple late payments detected. Use auto-pay.")
elif late_90 > 0:
    tips.append("⚠️ Avoid late payments using reminders.")

# Risk level
if risk > 70:
    tips.append("🚨 Avoid new credit until situation improves.")
elif risk > 40:
    tips.append("⚠️ Improve repayment consistency.")
else:
    tips.append("✅ Maintain current financial habits.")

if len(tips) == 0:
    tips.append("✅ Financial profile is strong.")

for i, tip in enumerate(tips, 1):
    st.write(f"{i}. {tip}")

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
    "Model": [mode]
})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    "📄 Download Report",
    csv,
    "risk_report.csv",
    "text/csv",
    key="download_unique_btn"
)
