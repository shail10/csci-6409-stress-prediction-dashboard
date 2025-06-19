# 🧠 Mental Health Insights and Prediction System (MHIPS)

This project uses machine learning to predict stress levels in individuals based on behavioral, demographic, and historical mental health data. Developed as a course project for **CSCI 6409 - Advanced Topics in Machine Learning** at Dalhousie University.

---

## 📊 Project Summary

- **Dataset**: Mental Health Dataset from Kaggle (292,364 records, 17 features)
- **Goal**: Predict self-reported growing stress (Yes, No, Maybe)
- **Models**: Random Forest, XGBoost, Decision Tree, KNN, Logistic Regression
- **Techniques**:
  - Feature selection via Chi-Square and Mutual Information
  - Hyperparameter tuning with `RandomizedSearchCV`
  - Counterfactual explanations using DiCE
  - Interpretability via feature importance plots

---

## 🔍 Key Findings

- **Top Predictors**: Days Indoors, Mood Swings, Work Interest, Occupation
- **Best Models**:  
  - 🏆 XGBoost: 99.07% accuracy  
  - 🏆 Random Forest: 99.04% accuracy  
- **Counterfactuals**: Identified actionable feature changes for intervention

---

## 🧠 Methodology

- Preprocessing: Encoding, feature selection, class balancing
- Classification: Multi-class setup (`Growing Stress = Yes | No | Maybe`)
- Interpretability: Feature importance & counterfactual generation
- Evaluation: Accuracy, RMSE, precision, recall, F1-score

---

## 📈 Visuals & Results

Plots include:
- Distribution & EDA visualizations
- Feature importance across models
- Learning curves
- Model comparison (boxplots)
- Counterfactual change frequencies

> 📌 *Due to self-reported survey nature, results are correlational and not diagnostic.*

---

## 👩‍💻 Authors

- **Yash Dinesh Harjani** – ys959012@dal.ca  
- **Shail Rajeshbhai Kardani** – sh475913@dal.ca  
- **Akshita Shyam Mendon** – ak621149@dal.ca

---

## 📁 Repo Structure

- `main.tex` – LaTeX source
- `references.bib` – BibTeX bibliography
- `*.png` – Visual assets for analysis (EDA, model eval, feature importance)
- `code/` – Python scripts for model training, tuning, and plots (if applicable)

---

## 📝 License

For educational use only. Dataset sourced from [Kaggle](https://www.kaggle.com/).
