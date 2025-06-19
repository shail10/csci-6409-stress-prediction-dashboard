# 🧠 Mental Health Insights and Prediction System (MHIPS)

A machine learning-based system to analyze and predict perceived stress levels from behavioral, demographic, and historical mental health data. Built as a term project for **CSCI 6409 – Advanced Topics in Machine Learning** at Dalhousie University.

---

## 📌 Project Overview

We designed **MHIPS** to predict increasing stress in individuals using real-world survey data. The goal was not just high accuracy, but also interpretability and real-world applicability. Our workflow includes data cleaning, feature selection, model training, evaluation, and counterfactual analysis.

---

## 🧑‍💻 What We Did

### 📁 Data Handling & Preprocessing
- **Dataset**: Kaggle Mental Health dataset with 292,364 records and 17 features
- Cleaned missing/incomplete records
- Encoded categorical features using one-hot encoding
- Removed irrelevant features like `timestamp`, `country`, `care options`, etc.
- Performed **Chi-Square** and **Mutual Information** tests to select the most impactful predictors

### 📊 Exploratory Data Analysis (EDA)
- Visualized class distributions and feature correlations
- Created plots for:
  - Days Indoors vs. Stress
  - Mood Swings
  - Gender-based stress reporting
  - Feature correlation heatmaps

### 🤖 Model Training
- **Models Used**:
  - Random Forest
  - XGBoost
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Logistic Regression (baseline)

- Implemented a utility function `tuneModelAndSummarize()` to automate:
  - Model initialization
  - Hyperparameter tuning via `RandomizedSearchCV`
  - Evaluation and scoring across multiple folds

### 🧪 Evaluation
- Compared models on:
  - Accuracy
  - RMSE
  - Precision, Recall, F1-Score
- Created learning curves and boxplots of cross-validation scores
- Random Forest and XGBoost performed best (accuracy ~99%)

### 📈 Feature Importance
- Extracted and visualized feature importances across models
- Top features:
  - Days Indoors
  - Mood Swings
  - Work Interest
  - Changes in Habits
- Verified model interpretations using statistical significance tests

### 🔍 Counterfactual Analysis
- Used **DiCE (Diverse Counterfactual Explanations)** to explain predictions
- Generated 2 counterfactuals per sample for 2,000 test instances
- Identified the most frequently changed features to alter stress predictions
- This provided **actionable insights** into how lifestyle changes could reduce perceived stress

---

## 📊 Results Snapshot

| Model              | Accuracy | Top Features                     |
|-------------------|----------|----------------------------------|
| XGBoost           | 99.07%   | Days Indoors, Work Interest      |
| Random Forest     | 99.04%   | Days Indoors, Mood Swings       |
| Decision Tree     | 98.89%   | Days Indoors, Changes Habits    |
| Logistic Regression | 43.87% | (Used as a weak baseline)        |

---

## 💡 Key Takeaways

- Behavioral factors (e.g., **isolation**, **emotional volatility**) are more predictive than static traits (e.g., family history)
- Counterfactuals enabled explainable ML by showing *what small changes could flip a prediction*
- A model isn't useful unless it's **interpretable**, **trustworthy**, and **actionable**

---

## 📘 Academic Info

- **Course**: CSCI 6409 – Process of Data Science 
- **Institution**: Dalhousie University  
- **Term**: Winter 2025

---

## 📜 License

This project was built for academic purposes. Dataset credits to [Kaggle Mental Health Survey](https://www.kaggle.com/). No clinical or medical use intended.
