import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/GiveMeSomeCredit.csv")
    df.fillna(df.median(), inplace=True)
    return df

df = load_data()

# =========================
# TRAIN MODEL (NO FILE NEEDED)
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
# UI
# =========================
st.title("💳 Credit Risk Prediction")

revolving = st.slider("Revolving Utilization", 0.0, 1.0, 0.5)
age = st.number_input("Age", 18, 100, 30)

total_debt = st.number_input("Total Debt", 0.0, 100000.0, 20000.0)
income = st.number_input("Income", 1.0, 100000.0, 30000.0)

debt = total_debt / income
st.write(f"Debt Ratio: {debt:.2f}")

late_90 = st.slider("90 Days Late", 0, 10, 0)

# =========================
# PREDICTION
# =========================
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

    st.success(f"Risk: {prob*100:.2f}%")
