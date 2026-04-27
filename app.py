import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>💳 Credit Risk Dashboard</h1>",
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
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model():
    X = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
    y = df["SeriousDlqin2yrs"]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    model.fit(X, y)
    return model, X.columns

model, feature_names = train_model()

# =========================
# LAYOUT (2 COLUMNS)
# =========================
col1, col2 = st.columns([1, 1])

# =========================
# INPUT PANEL
# =========================
with col1:
    st.subheader("📥 Customer Details")

    revolving = st.slider("Revolving Utilization", 0.0, 1.0, 0.5)
    age = st.number_input("Age", 18, 100, 30)

    total_debt = st.number_input("Total Debt", min_value=0.0, value=20000.0, step=1000.0)
    income = st.number_input("Income", min_value=1.0, value=30000.0, step=1000.0)

    if income <= 0:
        st.error("Income must be > 0")
        st.stop()

    debt = total_debt / income
    st.write(f"📊 Debt Ratio: **{debt:.2f}**")

    late_90 = st.slider("90 Days Late", 0, 10, 0)

    predict = st.button("🚀 Predict Risk")

# =========================
# OUTPUT PANEL
# =========================
with col2:

    if predict:

        input_df = pd.DataFrame([{
            "RevolvingUtilizationOfUnsecuredLines": revolving,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": debt,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": 5,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 1
        }])

        input_df = input_df[feature_names]

        prob = model.predict_proba(input_df)[0][1]
        risk_percent = prob * 100

        st.subheader("📊 Risk Score")

        # Risk meter
        st.progress(prob)

        if risk_percent >= 75:
            st.error(f"🚨 VERY HIGH RISK ({risk_percent:.1f}%)")
        elif risk_percent >= 50:
            st.warning(f"⚠️ HIGH RISK ({risk_percent:.1f}%)")
        elif risk_percent >= 30:
            st.info(f"⚠️ MODERATE RISK ({risk_percent:.1f}%)")
        else:
            st.success(f"✅ LOW RISK ({risk_percent:.1f}%)")

        # =========================
        # 📈 RISK BAR CHART
        # =========================
        st.subheader("📈 Risk Visualization")

        fig, ax = plt.subplots()
        ax.bar(["Safe", "Risk"], [1-prob, prob])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # =========================
        # 📊 FEATURE IMPORTANCE
        # =========================
        st.subheader("📊 Feature Importance")

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(5)

        fig2, ax2 = plt.subplots()
        ax2.barh(feat_df["Feature"], feat_df["Importance"])
        ax2.invert_yaxis()
        st.pyplot(fig2)

        # =========================
        # 🧠 INSIGHTS
        # =========================
        st.subheader("🧠 Key Insights")

        if debt > 0.5:
            st.write("⬆️ High debt ratio increases risk")

        if late_90 > 0:
            st.write("⬆️ Late payments increase risk")

        if income < 20000:
            st.write("⬆️ Low income increases risk")

        if revolving > 0.8:
            st.write("⬆️ High credit usage increases risk")

        # =========================
        # 💡 SUGGESTIONS
        # =========================
        st.subheader("💡 Recommendations")

        if risk_percent >= 50:
            st.write("👉 Reduce debt levels")
            st.write("👉 Avoid late payments")
            st.write("👉 Improve repayment history")
        else:
            st.write("👉 Maintain current financial discipline")

        st.success("✅ Analysis complete")
