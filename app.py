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
    return model, list(X.columns)

model, feature_names = train_model()

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

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
        st.error("❌ Income must be greater than 0")
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

        # =========================
        # SAFE INPUT BUILD
        # =========================
        input_dict = {
            "RevolvingUtilizationOfUnsecuredLines": revolving,
            "age": age,
            "DebtRatio": debt,
            "MonthlyIncome": income,
            "NumberOfTimes90DaysLate": late_90
        }

        safe_input = {}
        for col in feature_names:
            safe_input[col] = input_dict.get(col, 0)

        input_df = pd.DataFrame([safe_input])

        # =========================
        # MODEL PREDICTION
        # =========================
        prob = model.predict_proba(input_df)[0][1]
        safe_prob = float(prob)
        safe_prob = max(0.0, min(1.0, safe_prob))

        risk_percent = safe_prob * 100

        st.subheader("📊 Risk Score")
        st.progress(safe_prob)

        if risk_percent >= 75:
            st.error(f"🚨 VERY HIGH RISK ({risk_percent:.1f}%)")
        elif risk_percent >= 50:
            st.warning(f"⚠️ HIGH RISK ({risk_percent:.1f}%)")
        elif risk_percent >= 30:
            st.info(f"⚠️ MODERATE RISK ({risk_percent:.1f}%)")
        else:
            st.success(f"✅ LOW RISK ({risk_percent:.1f}%)")

        # =========================
        # 🧠 MODEL-BASED EXPLANATION
        # =========================
        st.subheader("🧠 Key Drivers (Model-Based)")

        importances = model.feature_importances_
        input_array = input_df.values[0]

        contributions = input_array * importances

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions
        })

        contrib_df["Abs"] = np.abs(contrib_df["Contribution"])
        contrib_df = contrib_df.sort_values(by="Abs", ascending=False)

        top = contrib_df.head(3)

        for _, row in top.iterrows():
            if row["Contribution"] > 0:
                st.write(f"⬆️ **{row['Feature']}** increased risk")
            else:
                st.write(f"⬇️ **{row['Feature']}** reduced risk")

        # =========================
        # 📊 SINGLE CLEAN CHART
        # =========================
        st.subheader("📊 Top Feature Impact")

        fig, ax = plt.subplots()

        top_plot = contrib_df.head(5)

        ax.barh(top_plot["Feature"], top_plot["Contribution"])
        ax.invert_yaxis()

        st.pyplot(fig)

        st.success("✅ Analysis complete")
