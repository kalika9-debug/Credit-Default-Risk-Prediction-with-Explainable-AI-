# 💳 Credit Default Risk Prediction with Explainable AI (XGBoost + SHAP)

> Predict loan default risk — and explain *why* it happens.

---

## 🚀 Overview

In real-world finance, predicting defaults isn’t enough — decisions must be explainable.

This project builds a machine learning model to:
- Predict whether a customer is likely to default
- Provide insights into *why* the prediction was made using SHAP

---

## 🎯 Key Features

✔ Predicts default probability  
✔ Uses multiple models and selects the best  
✔ Explains predictions using SHAP  
✔ Interactive Streamlit app  

---

## 📊 Dataset

**GiveMeSomeCredit (Kaggle)**

Includes:
- Credit usage
- Debt ratio
- Income
- Past payment delays

---

## ⚙️ Models Compared

- Logistic Regression (baseline)
- Decision Tree
- **XGBoost (best performer)**

---

## 📈 Evaluation Metrics

Used multiple metrics (not just accuracy):

- Precision  
- Recall  
- F1 Score  
- **AUC (primary metric)**  

👉 XGBoost achieved the best AUC and handled imbalance better.

---

## 🧠 Explainability (SHAP)

Instead of a black-box model, SHAP helps explain predictions.

### Insights:

- 📈 Higher **debt ratio** → increases risk  
- 📉 Lower **income** → higher default probability  
- ⏱️ Past **late payments** → strongest indicator  

---

## 🌐 Live App

👉 *(Add your Streamlit link here)*  

Features:
- Input customer details  
- Get risk percentage  
- Simple, fast predictions  

---

## 🛠️ Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  

---

## 📁 Project Structure
credit-default-prediction/
│
├── app.py
├── requirements.txt
├── README.md

---

## ⚠️ Limitations

- Trained on a subset of data  
- Limited hyperparameter tuning  

---

## 💡 Future Improvements

- Full dataset training  
- Hyperparameter optimization  
- SHAP visual dashboards  
- UI enhancement  

---

## 👩‍💻 About Me

**Kalika Tambat**  
BSc Data Science student exploring ML and real-world problem solving.

---

## ⭐ Final Thought

Machine learning shouldn’t just predict — it should explain.  
This project is a step toward building **trustworthy AI systems**.
