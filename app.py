import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("💳 Credit Risk Predictor (Internship Level)")
st.write("XGBoost + Real Dataset + Safe ML")

# =========================
# LOAD & TRAIN MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("credit_data.csv")

        # =========================
        # CLEANING
        # =========================
        df.replace("NA", np.nan, inplace=True)
        df.fillna(df.median(), inplace=True)

        # Fix extreme values
        df["DebtRatio"] = df["DebtRatio"].clip(0, 3)

        # =========================
        # FEATURES (MATCH DATASET)
        # =========================
        features = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "DebtRatio",
            "NumberOfTimes90DaysLate"
        ]

        X = df[features]
        y = df["SeriousDlqin2yrs"]

        # =========================
        # SPLIT
        # =========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =========================
        # MODEL (XGBOOST)
        # =========================
        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=10,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        return model

    except Exception as e:
        return None

model = load_model()

# =========================
# UI INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Customer Details")

    revolving = st.slider(
        "Credit Utilization",
        0.0, 1.0, 0.3
    )

    age = st.number_input(
        "Age", 18, 100, 30
    )

    total_debt = st.number_input(
        "Total Debt", value=20000.0
    )

    income = st.number_input(
        "Income", value=30000.0
    )

    if income <= 0:
        st.error("Income must be > 0")
        st.stop()

    # debt ratio
    debt_ratio = min(total_debt / income, 3)

    st.write(f"Debt Ratio: {debt_ratio:.2f}")

    late_90 = st.slider(
        "90 Days Late",
        0, 10, 0
    )

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
# PREDICT
# =========================
def predict():
    if model is not None:
        try:
            X = np.array([[
                revolving,
                age,
                debt_ratio,
                late_90
            ]])

            prob = model.predict_proba(X)[0][1]

            # =========================
            # BUSINESS RULES (IMPORTANT)
            # =========================
            if debt_ratio > 2.5:
                prob = max(prob, 0.7)

            if late_90 >= 3:
                prob = max(prob, 0.8)

            return prob, "XGBoost"

        except:
            return simple_model(), "Fallback"

    else:
        return simple_model(), "Fallback"

# =========================
# OUTPUT
# =========================
with col2:
    prob, mode = predict()
    risk = prob * 100

    st.subheader("📊 Risk Score")
    st.caption(f"Model Used: {mode}")

    st.progress(prob)

    if risk > 70:
        st.error(f"🚨 HIGH RISK ({risk:.1f}%)")
    elif risk > 40:
        st.warning(f"⚠️ MEDIUM RISK ({risk:.1f}%)")
    else:
        st.success(f"✅ LOW RISK ({risk:.1f}%)")

    # =========================
# SMART RISK REDUCTION ENGINE
# =========================
st.subheader("🧠 Personalized Risk Reduction Plan")

tips = []

# =========================
# UTILIZATION
# =========================
if revolving > 0.8:
    tips.append("🚨 Your credit utilization is extremely high. Pay down balances immediately.")
    tips.append("💳 Avoid using credit cards until utilization drops below 30%.")
elif revolving > 0.5:
    tips.append("⚠️ Reduce credit usage. Try to keep it under 30% for better score.")
elif revolving > 0.3:
    tips.append("ℹ️ Slightly high utilization. Monitor spending closely.")

# =========================
# DEBT RATIO
# =========================
if debt_ratio > 2.5:
    tips.append("🚨 Debt ratio is critically high. Consider restructuring loans.")
    tips.append("💰 Increase income sources or aggressively reduce liabilities.")
elif debt_ratio > 1:
    tips.append("⚠️ High debt ratio. Try reducing EMIs or avoid new loans.")
elif debt_ratio > 0.5:
    tips.append("ℹ️ Moderate debt. Maintain balance between income and expenses.")

# =========================
# LATE PAYMENTS
# =========================
if late_90 >= 3:
    tips.append("🚨 Frequent late payments detected. This severely impacts creditworthiness.")
    tips.append("📅 Set auto-debit for EMIs to avoid future delays.")
elif late_90 > 0:
    tips.append("⚠️ Avoid late payments. Even 1 delay affects your profile.")
    tips.append("🔔 Use reminders or auto-pay features.")

# =========================
# AGE / HISTORY
# =========================
if age < 25:
    tips.append("📈 Build credit history using small, timely repayments.")
elif age > 60:
    tips.append("🏦 Maintain stable financial behavior to preserve credit reliability.")

# =========================
# RISK LEVEL BASED ADVICE
# =========================
if risk > 75:
    tips.append("🚨 High risk profile: Avoid taking new loans until metrics improve.")
    tips.append("📉 Focus on reducing debt and improving repayment consistency.")
elif risk > 40:
    tips.append("⚠️ Moderate risk: Improve utilization and payment discipline.")
else:
    tips.append("✅ Low risk: Continue maintaining healthy financial habits.")

# =========================
# FALLBACK (NO ISSUES)
# =========================
if len(tips) == 0:
    tips.append("✅ Your financial profile is strong. Keep maintaining discipline.")

# =========================
# DISPLAY NICELY
# =========================
for i, tip in enumerate(tips, 1):
    st.write(f"{i}. {tip}")

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    if model is not None:
        st.subheader("📊 Feature Importance")

        importance = model.feature_importances_

        df_imp = pd.DataFrame({
            "Feature": [
                "Utilization",
                "Age",
                "Debt Ratio",
                "Late Payments"
            ],
            "Importance": importance
        }).sort_values(by="Importance")

        fig, ax = plt.subplots()
        ax.barh(df_imp["Feature"], df_imp["Importance"])

        st.pyplot(fig)

    # =========================
    # DEBUG
    # =========================
    st.write("DEBUG:", {
        "revolving": revolving,
        "age": age,
        "debt_ratio": debt_ratio,
        "late_90": late_90
    })

    # =========================
    # DOWNLOAD REPORT
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
