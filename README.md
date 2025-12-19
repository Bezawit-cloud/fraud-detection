# Fraud Detection & Credit Risk Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-green)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.11-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 🚀 Project Overview
This repository contains **preprocessing and analysis work** for two datasets:

1. **Fraud_Data.csv** – E-commerce fraud transaction data  
2. **Credit_Data.csv** – Credit dataset for default prediction  

**Objective:** Prepare clean, feature-rich datasets ready for **machine learning modeling**, including feature engineering, data transformation, and class imbalance handling.

---

## 🔍 Task 1 – Data Analysis & Preprocessing

### 1️⃣ Initial Observations

**Fraud Data**
- ✅ No missing values → clean dataset  
- ⚠️ Target highly imbalanced → requires special handling  
- 💰 `Amount` is skewed → consider log-transform or scaling  
- 📊 `V1–V28` are anonymized PCA features → good for modeling  
- ⏱ `Time` can be used for temporal features  

**Credit Data**
- ✅ No missing values  
- ⚠️ Extremely imbalanced target (0 → 226,602; 1 → 378)  
- 🔢 Most features numeric → ready for scaling  

---

### 2️⃣ Exploratory Data Analysis (EDA)

**Fraud Data Insights**

**Demographics & Browser (Categorical)**  
- **Sex:** More Males than Females → check for bias  
- **Browser:** Chrome dominates → browser-specific behavior might indicate fraud  

**Purchase Behavior (Continuous)**  
- `Purchase Value` skewed toward $30; rare high-value purchases → log-transform or scaling recommended  
- `Time` → can engineer `hour_of_day` and `day_of_week`  

**Target Variable Imbalance**  
- Class 0 → 90.6%, Class 1 → 9.3% → accuracy not reliable  
- Models need **Precision, Recall, F1-Score**  
- Use **SMOTE** or specialized loss functions  

**Summary Table of Key Insights**

| Feature        | Observation                     | Data Science Implication                                   |
|----------------|---------------------------------|-----------------------------------------------------------|
| Sex            | More Males than Females         | Check demographic bias in target outcomes                |
| Browser        | Chrome Dominance                | Browser-specific behavior might be a strong feature      |
| Purchase Value | Skewed towards $30              | Consider log-transform or scaling for models             |
| Target         | 90% vs 9% imbalance             | Requires SMOTE or specialized loss functions            |

---

### 3️⃣ Feature Engineering

**Fraud Data**
- Transaction frequency per user  
- Time-based: `hour_of_day`, `day_of_week`  
- Duration since signup: `time_since_signup`  

**Credit Data**
- Ratio feature: `credit_per_month = credit_amount / duration`  
- Scaled numeric features: `credit_amount`, `age`, `duration`  

---

### 4️⃣ Data Transformation
- **Numeric scaling:** `StandardScaler` applied to numeric features  
- **Categorical encoding:** One-Hot Encoding applied (Credit Data only)  

---

### 5️⃣ Handling Class Imbalance

| Dataset          | Before SMOTE          | After SMOTE                            |
|-----------------|---------------------|---------------------------------------|
| Fraud_Data.csv  | 0 → 90.6%, 1 → 9.3% | 0 → 100%, 1 → 100% (balanced train)  |
| Credit_Data.csv | 0 → 226,602, 1 → 378| 0 → 226,602, 1 → 226,602 (balanced)  |

**Justification:**  
- SMOTE synthetically increases minority class without losing majority data  
- Ensures model can learn patterns for **rare events** (fraud/default)  

---

### 6️⃣ Dataset Ready for Modeling
- **Training set:** Balanced (`X_res`, `y_res`)  
- **Test set:** Original, imbalanced (`X_test`, `y_test`)  
- Features scaled and encoded  

---

### 7️⃣ Next Steps
- Train ML models (Logistic Regression, Random Forest, XGBoost)  
- Optimize hyperparameters  
- Evaluate using **Precision, Recall, F1-Score, AUC**  

---

### 8️⃣ Repository Files
- `Task1_Fraud_Data_Preprocessing.ipynb` – Fraud dataset preprocessing  
- `Task1_Credit_Data_Preprocessing.ipynb` – Credit dataset preprocessing  
- `README.md` – Project overview and workflow  
- `.gitignore` – Ignored files (datasets, virtual environments, checkpoints)  

---

### 💡 Notes
- Large datasets are **not included**; download links/instructions are in notebooks  
- Preprocessing forms the foundation for **fraud detection and credit risk modeling**  

---

**Author:** Your Name  
**Date:** 2025
