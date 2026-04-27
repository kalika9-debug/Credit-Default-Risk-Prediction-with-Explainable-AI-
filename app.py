import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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
    return model

model = train_model()

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Enter Customer Details")

revolving = st.slider("Revolving Utilization (0–1)", 0.0, 1.0, 0.5)
age = st.number_input("Age", 18, 100, 30)

# SAFE INPUTS (NO LIMIT)
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
    # 📊 RISK METER
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
    # 🧠 SIMPLE EXPLANATION
    # =========================
    st.subheader("🧠 Key Risk Factors")

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
        st.write("👉 Maintain financial discipline")
        st.write("👉 Keep debt ratio low")

    st.markdown("---")
    st.success("✅ Analysis completed")
