import streamlit as st
import numpy as np
import pickle
import shap

# Load model
model = pickle.load(open("model.pkl", "rb"))

@st.cache_resource
def get_explainer():
    return shap.Explainer(model)

explainer = get_explainer()

# UI
st.title("💳 Credit Default Risk Prediction")

revolving = st.slider("Revolving Utilization", 0.0, 1.0, 0.5)
age = st.number_input("Age", 18, 100, 30)

st.subheader("💡 Debt Ratio Calculator")

total_debt = st.number_input("Total Debt", 0.0, 1000000.0, 20000.0)
income = st.number_input("Income", 1.0, 1000000.0, 30000.0)

debt = total_debt / income
st.write(f"Debt Ratio: {debt:.2f}")

late_90 = st.slider("90 Days Late", 0, 10, 0)

# Predict
if st.button("Predict"):

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
    risk = prob * 100

    st.subheader("📊 Risk Score")

    if risk >= 75:
        st.error(f"VERY HIGH RISK ({risk:.1f}%)")
    elif risk >= 50:
        st.warning(f"HIGH RISK ({risk:.1f}%)")
    elif risk >= 30:
        st.info(f"MODERATE RISK ({risk:.1f}%)")
    else:
        st.success(f"LOW RISK ({risk:.1f}%)")

    # SHAP
    shap_values = explainer(input_data)

    st.subheader("🧠 Explanation")

    features = [
        "Revolving","Age","30-59","Debt",
        "Income","Credit","90 Late","Real Estate","60-89","Dependents"
    ]

    impacts = shap_values.values[0]
    idx = np.argsort(np.abs(impacts))[::-1]

    for i in idx[:3]:
        if impacts[i] > 0:
            st.write(f"⬆️ {features[i]} increases risk")
        else:
            st.write(f"⬇️ {features[i]} reduces risk")
