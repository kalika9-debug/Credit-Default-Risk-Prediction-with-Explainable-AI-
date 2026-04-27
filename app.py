import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk App", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>💳 Credit Risk Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

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
# TRAIN MODEL (NO PIPELINE)
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

    # 🔥 IMPORTANT: remove feature-name dependency
    model.fit(X.values, y.values)

    return model, list(X.columns)

model, feature_names = train_model()

# =========================
# UI LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUT SECTION
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

    debt_ratio = np.clip(total_debt / income, 0, 5)

    st.markdown(f"""
    <div style="padding:10px;border-radius:10px;background:#111;color:white;">
        Debt Ratio: <b>{debt_ratio:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# BUILD INPUT (STRICT ORDER)
# =========================
input_dict = {
    "RevolvingUtilizationOfUnsecuredLines": revolving,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": income,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
}

# Convert to correct order
input_list = [input_dict.get(col, 0) for col in feature_names]
input_array = np.array([input_list])

# =========================
# OUTPUT SECTION
# =========================
with col2:
    st.subheader("📊 Risk Score")

    prob = model.predict_proba(input_array)[0][1]
    prob = float(np.clip(prob, 0, 1))
    risk = prob * 100

    st.progress(prob)

    if risk >= 75:
        st.error(f"🚨 VERY HIGH RISK ({risk:.1f}%)")
    elif risk >= 50:
        st.warning(f"⚠️ HIGH RISK ({risk:.1f}%)")
    elif risk >= 30:
        st.info(f"⚠️ MODERATE RISK ({risk:.1f}%)")
    else:
        st.success(f"✅ LOW RISK ({risk:.1f}%)")

    # =========================
    # FEATURE IMPORTANCE CHART
    # =========================
    st.subheader("📊 Key Drivers")

    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
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
    report = pd.DataFrame([input_dict])
    report["Risk (%)"] = risk

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📄 Download Report",
        csv,
        "credit_risk_report.csv",
        "text/csv"
    )
