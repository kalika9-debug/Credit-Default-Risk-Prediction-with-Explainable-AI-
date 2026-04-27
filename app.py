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
# TRAIN MODEL
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
    return model, X

model, X = train_model()
feature_names = list(X.columns)

# =========================
# UI LAYOUT
# =========================
col1, col2 = st.columns([1, 1])

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

    debt_ratio = np.clip(total_debt / income, 0, 5)
    income_scaled = np.log1p(income)

    st.markdown(f"""
    <div style="padding:10px;border-radius:10px;background-color:#111;">
        <b>Debt Ratio:</b> {debt_ratio:.2f}
    </div>
    """, unsafe_allow_html=True)

    late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# OUTPUT
# =========================
with col2:

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

    input_df = pd.DataFrame([input_dict])

    # =========================
    # PREDICTION
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
    # SHAP (SAFE VERSION)
    # =========================
    st.subheader("🧠 Model Explanation (SHAP)")

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

        # show top 3
        for _, row in shap_df.head(3).iterrows():
            direction = "⬆️ increases risk" if row["Impact"] > 0 else "⬇️ reduces risk"
            st.write(f"{direction}: **{row['Feature']}**")

        # =========================
        # SINGLE CLEAN CHART
        # =========================
        st.subheader("📊 SHAP Impact")

        fig, ax = plt.subplots()
        top = shap_df.head(5)

        ax.barh(top["Feature"], top["Impact"])
        ax.invert_yaxis()

        st.pyplot(fig)

    except:
        st.warning("SHAP explanation unavailable")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.subheader("📄 Download Report")

    report = input_df.copy()
    report["Risk (%)"] = risk_percent

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Report",
        data=csv,
        file_name="credit_risk_report.csv",
        mime="text/csv"
    )
