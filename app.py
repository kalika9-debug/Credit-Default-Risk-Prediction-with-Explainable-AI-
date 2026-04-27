import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")

st.title("💳 Credit Risk Intelligence")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cs-training.csv")
    df.fillna(df.median(), inplace=True)
    return df

df = load_data()

# =========================
# TRAIN MODEL (NO PIPELINE → NO ERROR)
# =========================
@st.cache_resource
def train_model():
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    model.fit(X, y)
    return model, X

model, X = train_model()

# =========================
# UI
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
    debt_ratio = np.clip(debt_ratio, 0, 5)

    st.write(f"Debt Ratio: {debt_ratio:.2f}")

    late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# SAFE INPUT (KEY FIX)
# =========================
# 👉 Start from REAL DATA ROW
input_df = X.iloc[[0]].copy()

# 👉 Replace only what user controls
input_df["RevolvingUtilizationOfUnsecuredLines"] = revolving
input_df["age"] = age
input_df["DebtRatio"] = debt_ratio
input_df["MonthlyIncome"] = income
input_df["NumberOfTimes90DaysLate"] = late_90

# =========================
# PREDICTION
# =========================
with col2:
    prob = model.predict_proba(input_df)[0][1]
    prob = float(np.clip(prob, 0, 1))
    risk = prob * 100

    st.subheader("📊 Risk Score")

    st.progress(prob)

    if risk > 70:
        st.error(f"🚨 High Risk ({risk:.1f}%)")
    elif risk > 40:
        st.warning(f"⚠️ Medium Risk ({risk:.1f}%)")
    else:
        st.success(f"✅ Low Risk ({risk:.1f}%)")

    # =========================
    # SIMPLE FEATURE IMPORTANCE (SAFE)
    # =========================
    st.subheader("📊 Key Factors")

    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    top = imp_df.head(5)

    ax.barh(top["Feature"], top["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)

    # =========================
    # DOWNLOAD REPORT
    # =========================
    report = input_df.copy()
    report["Risk"] = risk

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Report",
        csv,
        "risk_report.csv",
        "text/csv"
    )
