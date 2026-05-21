# 🔥 Boiler Steam Temperature Predictor

A machine learning web app for time-series prediction of **boiler output steam temperature** (`TE_8332A.AV_0#`) using industrial sensor data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://boiler-time-series-5pq2bxnvjhbmgbstuexc6d.streamlit.app))

---

## 📌 Problem Statement

Industrial boilers generate steam at precise temperatures. Predicting output steam temperature from upstream sensor readings helps operators anticipate deviations, reduce energy waste, and prevent equipment failure.

## 📊 Dataset

- Industrial boiler time-series sensor data
- ~20+ sensor readings (pressure, temperature, airflow, etc.)
- Target: `TE_8332A.AV_0#` — Boiler Output Steam Temperature

## 🧠 Models Implemented

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Ensemble tree method |
| XGBoost | Gradient boosted trees |
| **Hybrid (Linear + XGBoost)** | **Best performance** |

> 🏆 **Best Result (Hybrid, no outliers):** MAE: 2.32 · RMSE: 3.00 · R²: 0.36

## 🔧 Pipeline

```
Raw CSV → Null Imputation → Standard Scaling → Feature Selection
       → Outlier Removal (Z-score) → Train/Test Split (80/20) → Model → Metrics + Plot
```

**Optional:** Lasso-based feature selection

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/boiler-time-series.git
cd boiler-time-series
pip install -r requirements.txt
streamlit run app.py
```



## 📁 Project Structure

```
boiler-time-series/
├── app.py                    # Streamlit web app
├── Boiler_Time_Series.ipynb  # Full analysis notebook
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Scikit-learn` · `XGBoost` · `Pandas` · `Seaborn` · `SciPy`

---

*Built as part of an industrial ML project on boiler performance forecasting.*
