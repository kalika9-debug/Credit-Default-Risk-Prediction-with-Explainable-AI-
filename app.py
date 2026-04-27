import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")

st.markdown("<h1 style='text-align:center;'>💳 Credit Risk Intelligence</h1>", unsafe_allow_html=True)
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
# TRAIN MODEL (PROPER PIPELINE)
# =========================
@st.cache_resource
def train_model():
    X = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
    y = df["SeriousDlqin2yrs"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        ))
    ])

    model.fit(X, y)
    return model, list(X.columns)

model, feature_names = train_model()

# =========================
# UI LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUTS
# =========================
with col1:
    st.subheader("📥 Customer Profile")

    revolving = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
    age = st.number_input("Age", 18, 100, 30)

    total_debt = st.number_input("Total Debt", min_value=0.0, value=20000.0, step=1000.0)
    income = st.number_input("Monthly Income", min_value=1.0, value=30000.0, step=1000.0)

    if income <= 0:
        st.error("Income must be > 0")
        st.stop()

    # =========================
    # ENGINEERED FEATURES (IMPORTANT)
    # =========================
    debt_ratio = total_debt / income
    debt_ratio = np.clip(debt_ratio, 0, 5)

    income_scaled = np.log1p(income)  # MUCH better than raw scaling

    st.write(f"📊 Debt Ratio: **{debt_ratio:.2f}**")

    late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# LIVE PREDICTION (NO BUTTON)
# =========================
with col2:

    # =========================
    # BUILD INPUT (CONSISTENT)
    # =========================
    input_dict = {
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income_scaled,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }

    safe_input = {col: input_dict.get(col, 0) for col in feature_names}
    input_df = pd.DataFrame([safe_input])

    # =========================
    # MODEL PREDICTION
    # =========================
    prob = model.predict_proba(input_df)[0][1]
    prob = float(np.clip(prob, 0, 1))
    risk_percent = prob * 100

    st.subheader("📊 Risk Score")
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
    # MODEL-BASED EXPLANATION
    # =========================
    st.subheader("🧠 Key Drivers")

    xgb_model = model.named_steps["xgb"]
    importances = xgb_model.feature_importances_

    input_array = input_df.values[0]
    contributions = input_array * importances

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": contributions
    })

    contrib_df["Abs"] = np.abs(contrib_df["Impact"])
    contrib_df = contrib_df.sort_values(by="Abs", ascending=False)

    for _, row in contrib_df.head(3).iterrows():
        direction = "⬆️ increases risk" if row["Impact"] > 0 else "⬇️ reduces risk"
        st.write(f"{direction}: **{row['Feature']}**")

    # =========================
    # SINGLE STRONG CHART
    # =========================
    st.subheader("📊 Top Feature Impact")

    fig, ax = plt.subplots()
    top = contrib_df.head(5)

    ax.barh(top["Feature"], top["Impact"])
    ax.invert_yaxis()

    st.pyplot(fig)
