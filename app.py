import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>💳 Credit Risk Intelligence</h1>
<p style='text-align:center;color:gray;'>AI-powered risk scoring with explainability</p>
""", unsafe_allow_html=True)

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

    pipeline = Pipeline([
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

    pipeline.fit(X, y)

    return pipeline, X.columns

model, feature_names = train_model()

# =========================
# UI LAYOUT
# =========================
col1, col2 = st.columns([1, 1])

# =========================
# INPUT SECTION
# =========================
with col1:
    st.subheader("📥 Customer Profile")

    revolving = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
    age = st.number_input("Age", 18, 100, 30)

    total_debt = st.number_input("Total Debt", min_value=0.0, value=20000.0)
    income = st.number_input("Monthly Income", min_value=1.0, value=30000.0)

    if income <= 0:
        st.error("Income must be > 0")
        st.stop()

    debt_ratio = np.clip(total_debt / income, 0, 5)
    income_scaled = np.log1p(income)

    st.markdown(f"""
    <div style="padding:12px;border-radius:10px;background-color:#111;">
        📊 Debt Ratio: <b>{debt_ratio:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# BUILD SAFE INPUT (KEY FIX)
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

# 🔥 IMPORTANT: MATCH TRAINING FEATURES
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0

for col in input_dict:
    if col in input_df.columns:
        input_df[col] = input_dict[col]

# =========================
# PREDICTION
# =========================
with col2:

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
    # SHAP EXPLANATION
    # =========================
    st.subheader("🧠 SHAP Explanation")

    try:
        xgb_model = model.named_steps["xgb"]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values[0]
        })

        shap_df["Abs"] = np.abs(shap_df["Impact"])
        shap_df = shap_df.sort_values(by="Abs", ascending=False)

        # Top drivers
        for _, row in shap_df.head(3).iterrows():
            direction = "⬆️ increases risk" if row["Impact"] > 0 else "⬇️ reduces risk"
            st.write(f"{direction}: **{row['Feature']}**")

        # Chart
        fig, ax = plt.subplots()
        top = shap_df.head(5)

        ax.barh(top["Feature"], top["Impact"])
        ax.invert_yaxis()

        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP failed — install compatible version")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.subheader("📄 Export Report")

    report = input_df.copy()
    report["Risk (%)"] = risk_percent

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Report",
        csv,
        "risk_report.csv",
        "text/csv"
    )
