import streamlit as st
import pandas as pd

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Credit Risk", layout="centered")

st.title("💳 Credit Risk Checker")
st.write("Simple and clear risk analysis")

# =========================
# INPUT
# =========================
st.subheader("Enter Details")

revolving = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
age = st.number_input("Age", 18, 100, 30)

debt = st.number_input("Total Debt", value=20000.0)
income = st.number_input("Income", value=30000.0)

late = st.slider("Late Payments (90 days)", 0, 10, 0)

if income <= 0:
    st.error("Income must be > 0")
    st.stop()

debt_ratio = debt / income
st.write(f"Debt Ratio: {debt_ratio:.2f}")

# =========================
# SIMPLE LOGIC (CLEAR)
# =========================
score = 0
reasons = []

# utilization
if revolving > 0.7:
    score += 40
    reasons.append("High credit utilization")
elif revolving > 0.4:
    score += 20
    reasons.append("Moderate credit utilization")

# debt ratio
if debt_ratio > 2:
    score += 30
    reasons.append("Very high debt compared to income")
elif debt_ratio > 1:
    score += 15
    reasons.append("High debt compared to income")

# late payments
if late >= 3:
    score += 30
    reasons.append("Multiple late payments")
elif late > 0:
    score += 15
    reasons.append("Some late payments")

# age factor
if age < 25:
    score += 10
    reasons.append("Short credit history")

# cap score
score = min(score, 100)

# =========================
# RESULT
# =========================
st.markdown("---")
st.subheader("📊 Risk Result")

st.metric("Risk Score", f"{score}%")

if score > 70:
    st.error("🚨 High Risk")
elif score > 40:
    st.warning("⚠️ Medium Risk")
else:
    st.success("✅ Low Risk")

# =========================
# WHY (IMPORTANT)
# =========================
st.subheader("📌 Why this risk?")

if reasons:
    for r in reasons:
        st.write(f"- {r}")
else:
    st.write("No major risk factors detected")

# =========================
# SUGGESTIONS (MATCH REASONS)
# =========================
st.subheader("🧠 What to improve")

tips = []

if "High credit utilization" in reasons or "Moderate credit utilization" in reasons:
    tips.append("Reduce credit usage below 30%")

if "Very high debt compared to income" in reasons or "High debt compared to income" in reasons:
    tips.append("Reduce debt or increase income")

if "Multiple late payments" in reasons or "Some late payments" in reasons:
    tips.append("Avoid late payments (use auto-pay)")

if "Short credit history" in reasons:
    tips.append("Build credit history with small timely payments")

if not tips:
    tips.append("Maintain current financial habits")

for t in tips:
    st.write(f"- {t}")

# =========================
# DOWNLOAD
# =========================
report = pd.DataFrame({
    "Risk Score (%)": [score],
    "Debt Ratio": [debt_ratio],
    "Late Payments": [late]
})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    "📄 Download Report",
    csv,
    "report.csv",
    "text/csv",
    key="simple_download"
)
