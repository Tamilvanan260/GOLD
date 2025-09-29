# 📊 Quantitative Analysis of Gold Prices using Machine Learning

## 🚀 Tools & Platforms
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/VS%20Code-0078D4?style=for-the-badge&logo=visualstudiocode&logoColor=white"/>
  <img src="https://img.shields.io/badge/Data%20Visualization-FF6F00?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-102230?style=for-the-badge&logo=scikitlearn&logoColor=F7931E"/>
</p>

---

## 📌 Project Overview
This project focuses on analyzing and predicting **historical gold price trends** using advanced **machine learning regression models**.  
Gold is influenced by multiple economic indicators, and accurate prediction helps investors, analysts, and policymakers make informed decisions.  

The goal is to **compare multiple regression models** and identify the best-performing one.

---

## 🎯 Objectives
- 🔹 Clean and preprocess gold price datasets (handle missing values, duplicates, scaling).  
- 🔹 Perform **Exploratory Data Analysis (EDA)** to uncover hidden patterns.  
- 🔹 Apply **feature selection techniques** (Mutual Information).  
- 🔹 Train and evaluate ML models:  
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  
  - Support Vector Regressor (SVR)  
- 🔹 Compare model performance using **Mean Squared Error (MSE)**.  
- 🔹 Derive insights into economic factors influencing gold prices (e.g., USD to INR exchange rate).  

---

## 🧩 Methodology
1. **Data Collection** → Historical gold price data from Kaggle/financial sources.  
2. **Data Preprocessing** → Handle missing values, remove duplicates, treat outliers, scale features.  
3. **Feature Selection** → Use Mutual Information to find most relevant predictors.  
4. **Model Training** → Implement multiple regression algorithms.  
5. **Evaluation** → Compare models using **MSE**.  
6. **Visualization** → Heatmaps, histograms, correlation plots, and bar charts.  

---

## 📊 Results
| Model                     | Mean Squared Error (MSE) | Remarks |
|----------------------------|--------------------------|---------|
| Linear Regression          | 48.72                   | Struggled with non-linear data |
| Decision Tree Regressor    | 28.65                   | Good for non-linear, but overfitting |
| Random Forest Regressor    | 19.34                   | Balanced accuracy |
| **Gradient Boosting** ✅    | **17.84**               | Best-performing model |
| Support Vector Regressor   | 29.87                   | Sensitive to hyperparameters |

✅ **Gradient Boosting Regressor** achieved the lowest error → most accurate model.  

---

## 📈 Visualizations
- 📌 Correlation Heatmap → feature relationships  
- 📌 Histograms → gold price distribution  
- 📌 Residual Plots → model errors  
- 📌 Bar Charts → comparing MSE values  

---

## 📂 Project Structure


## 👨‍💻 Author
**Tamilvanan I**  
B.Sc. Artificial Intelligence and Machine Learning  
Sri Krishna Adithya College of Arts and Science, Coimbatore  

📅 Project Completed: **March 2025**  

---

⭐ If you like this project, don’t forget to **star this repository** on GitHub!
