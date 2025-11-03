# 🏦  Loan Approval Prediction – AI-Powered Decision Engine

**Predicts whether a loan application will be *Approved* or *Rejected* using real-world banking features.**


---

## Project Overview

This project builds a **machine learning model** to predict loan approval status based on applicant data such as:

- CIBIL Score
- Annual Income
- Loan Amount & Term
- Assets (Residential, Commercial, Luxury, Bank)
- Education & Employment Type
- Number of Dependents

We trained **two models**:
1. **Logistic Regression** (interpretable )
2. **Decision Tree** (captures non-linear patterns)

Used **SMOTE** for class imbalance and **RandomizedSearchCV** for hyperparameter tuning.

The **best model is saved** with `joblib` and deployed via **Streamlit** as an interactive web app.

---

## Key Performance Metrics (Corrected from Classification Report)

| Model             | **Accuracy** | **F1-Score (Approved)** | **F1-Score (Rejected)** | **ROC-AUC** |
|-------------------|--------------|--------------------------|--------------------------|-------------|
| **Logistic Regression** | **0.93** | **0.94** | **0.90** | **0.9735** |
| **Decision Tree**       | **0.98** | **0.98** | **0.97** | **0.9790** |

> **Winner: Decision Tree** – Higher **Accuracy (0.98)** & **F1-Score (0.98)**
---
## Step-by-Step Workflow (Jupyter Notebook)

| Step | Description |
|------|-------------|
| **1. Import Libraries** | `pandas`, `numpy`, `sklearn`, `imblearn`, `matplotlib`, `seaborn`, `joblib` |
| **2. Load Data** | `pd.read_csv("loan_approval_dataset.csv")` |
| **3. EDA** | Checked missing values, class distribution, data types |
| **4. Visualization** | Boxplots, countplots, correlation heatmap |
| **5. Feature Engineering & Preprocessing** | <ul><li>`.str.strip()` on text columns (critical!)</li><li>One-Hot Encoding: `education`, `self_employed`</li><li>StandardScaler on numeric features</li></ul> |
| **6. Train-Test Split** | `stratify=y`, `test_size=0.2`, `random_state=42` |
| **7. Logistic Regression + SMOTE + Random Search** | <ul><li>Imbalanced pipeline</li><li>Tuned `C`, `penalty`, `solver`</li><li>Accuracy: **0.93**, F1(Approved): **0.94**</li></ul> |
| **8. Decision Tree + SMOTE + Random Search** | <ul><li>Tuned `max_depth`, `min_samples_split`, etc.</li><li>Accuracy: **0.98**, F1(Approved): **0.98**</li></ul> |
| **9. Model Comparison** | Decision Tree wins with **98% Accuracy** |
| **10. Feature Importance (Decision Tree)** | Plotted top 10 important features |
| **11. Pick Best Model** | **Decision Tree** (saved as `loan_approval_best.pkl`) |
| **12. Save Model with `joblib`** | `joblib.dump(best_dt, "loan_approval_best.pkl")` |
| **13. Deployment using Streamlit** | <ul><li>Created `app.py` with interactive UI</li><li>Loaded `loan_approval_best.pkl`</li><li>Real-time prediction with CIBIL penalty</li><li>Deployed locally via `streamlit run app.py`</li><li>Ready for **Streamlit Cloud** (free public URL)</li></ul> |
------
## 📂Project Structure
```plaintext

├── loan_approval_dataset.csv       # Raw dataset
├── loan_predictions.ipynb          # Full Jupyter Notebook (training + analysis)
├── app.py                          # Streamlit Web App
├── loan_approval_best.pkl          # Saved best model (Decision Tree)
├── requirements.txt                # Python dependencies
└── README.md                       # This file


