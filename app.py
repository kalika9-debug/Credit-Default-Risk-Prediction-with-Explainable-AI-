import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>💳 Credit Risk Prediction</h1>",
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
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    return model, X

model, X = train_model()

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Enter Customer Details")

revolving = st.slider("Revolving Utilization (0–1)", 0.0, 1.0, 0.5)
age = st.number_input("Age", 18, 100, 30)

# SAFE INPUTS (NO LIMITS)
total_debt = st.number_input(
    "Total Debt",
    min_value=0.0,
    value=20000.0,
    step=1000.0
)

income = st.number_input(
    "Income",
    min_value=1.0,
    value=30000.0,
    step=1000.0
)

# SAFETY CHECK
if income <= 0:
    st.error("❌ Income must be greater than 0")
    st.stop()

debt = total_debt / income

if debt > 10:
    st.warning("⚠️ Extremely high debt ratio")

st.write(f"📊 Debt Ratio: **{debt:.2f}**")

late_90 = st.slider("90 Days Late", 0, 10, 0)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Risk"):

    input_data = np.array([[ 
        revolving,
        age,
        0,
        debt,
        income,
        5,
        late_90,
        1,
        0,
        1
    ]])

    prob = model.predict_proba(input_data)[0][1]
    risk_percent = prob * 100

    # =========================
    # 🎯 RISK METER UI
    # =========================
    st.subheader("📊 Risk Level")

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
    # 🧠 SHAP EXPLANATION
    # =========================
    st.subheader("🧠 Why this prediction?")

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(input_data)

        feature_names = X.columns
        impacts = shap_values.values[0]

        sorted_idx = np.argsort(np.abs(impacts))[::-1]

        for i in sorted_idx[:3]:
            if impacts[i] > 0:
                st.write(f"⬆️ **{feature_names[i]}** increases risk")
            else:
                st.write(f"⬇️ **{feature_names[i]}** reduces risk")

    except:
        st.warning("⚠️ Explanation unavailable")

    # =========================
    # 💡 SMART SUGGESTIONS
    # =========================
    st.subheader("💡 Recommendations")

    if risk_percent >= 50:
        st.write("👉 Reduce debt levels")
        st.write("👉 Avoid late payments")
        st.write("👉 Improve repayment history")
    else:
        st.write("👉 Maintain current financial habits")
        st.write("👉 Keep debt ratio low")

    st.markdown("---")
    st.success("✅ Analysis completed")
