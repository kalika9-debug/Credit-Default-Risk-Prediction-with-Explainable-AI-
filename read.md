# 💳 Credit Default Risk Prediction with Explainable AI (XGBoost + SHAP)

> A simple but practical machine learning project to predict loan default risk — with explanations, not just predictions.

---

## 🚀 Overview

In real life, banks don’t just want to know *who* might default — they want to know *why*.

This project focuses on building a model that predicts whether a customer is likely to default on a loan, and more importantly, explains the reasoning behind that prediction using SHAP.

---

## 🎯 What this project does

* Predicts if a customer is risky or safe
* Highlights the key factors behind the prediction
* Makes the model more transparent and understandable

---

## 📊 Dataset

I used the **GiveMeSomeCredit dataset from Kaggle**, which includes:

* Income details
* Debt ratio
* Past payment behavior
* Other financial indicators

---

## ⚙️ Models I tried

Instead of jumping straight to one model, I compared a few:

* Logistic Regression (baseline)
* Decision Tree
* **XGBoost (performed the best)**

---

## 📈 How I evaluated the models

I didn’t rely on just accuracy. I used:

* Precision
* Recall
* F1 Score
* **AUC (most important here)**

👉 XGBoost clearly performed better, especially in identifying defaulters.

---

## 🧠 Explainability (SHAP)

This is where things get interesting.

Using SHAP, I was able to see:

* Which features matter the most
* How each feature affects the prediction

### Key takeaways:

* Higher **debt ratio** increases risk
* Lower **income** makes repayment harder
* Past **delinquencies** strongly signal future default

👉 This turns the model from a black box into something you can actually trust.

---

## 🌐 Deployment

I built a simple app using Streamlit where you can:

* Enter customer details
* Get instant predictions
* See the probability of default

To run locally:

```
streamlit run app.py
```

---

## 📁 Project Structure

```
credit-default-prediction/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
```

---

## ⚠️ Limitations

* The model was trained on a subset of data (due to system limits)
* Could be improved with more data and tuning

---

## 💡 What I’d improve next

* Train on the full dataset
* Tune hyperparameters
* Add SHAP visualizations directly into the app
* Improve the UI

---

## 👩‍💻 About me

**Kalika Tambat**
Student exploring data science, machine learning, and real-world problem solving.

---

## ⭐ Final note

This project is a step toward building machine learning systems that are not just accurate, but also **explainable and trustworthy**.
