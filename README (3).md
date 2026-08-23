# 🚕 Automatidata — NYC Taxi Fare & Tip Analysis
**Google Advanced Data Analytics Professional Certificate — Capstone Project**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Modeling-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Modeling-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 📌 Overview

This project was completed as the capstone for the **Google Advanced Data Analytics Professional Certificate**, framed as a real-world consulting engagement.

**Automatidata**, a data consulting firm, is working with the **New York City Taxi and Limousine Commission (TLC)** — the agency responsible for licensing and regulating NYC's taxi cabs and for-hire vehicles. TLC data covers over 200,000 licensed vehicles making roughly **one million trips per day**.

The engagement was broken into five sequential parts, each building on the last — moving from raw data inspection all the way to a deployable machine learning model:

| Part | Focus | Deliverable |
|---|---|---|
| **1** | Initial Data Exploration | Understand structure, types, and quality of the raw dataset |
| **2** | Exploratory Data Analysis | Visualize distributions, trends, and outliers (Python) |
| **3** | Hypothesis Testing | A/B test on payment type vs. fare amount |
| **4** | Regression Analysis | Multiple linear regression to predict taxi fares |
| **5** | Model Development | Classification model to predict generous tippers |

---

## 🎯 Business Problems

1. **Fare Prediction:** Build a regression model to estimate a taxi fare *before* a ride begins, using historical TLC trip data — enabling a rider-facing fare estimate feature.
2. **Tip Prediction:** Build a classification model to predict whether a rider is likely to be a **"generous tipper"** (tipping ≥ 20%), to help understand and potentially improve driver earnings and satisfaction.

---

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `Part_1_Initial_Data_Exploration.ipynb` | First look at the dataset: structure, types, nulls, outlier detection |
| `Part_1_Initial_Data_Exploration_executive_summary.pptx` | Stakeholder-facing summary of Part 1 findings |
| `Part_2_Exploratory_Data_Analysis.ipynb` | Full EDA — boxplots, histograms, trends by time/location/vendor |
| `Part_2_Exploratory_Data_Analysis_executive_summary.pptx` | Stakeholder-facing summary of Part 2 findings |
| `Part_3_Hypothesis_Testing.ipynb` | Two-sample t-test comparing fare amounts by payment type |
| `Part_3_Hypothesis_Testing_executive_summary.pptx` | Stakeholder-facing summary of Part 3 findings |
| `Part_4_Regression_Analysis.ipynb` | Feature engineering + multiple linear regression fare model |
| `Part_4_Regression_Analysis_executive_summary.pptx` | Stakeholder-facing summary of Part 4 findings |
| `Part_5_Model_Development.ipynb` | Classification models (Random Forest, XGBoost) to predict generous tippers |
| `Part_5_Model_Development_executive_summary.pptx` | Stakeholder-facing summary of Part 5 findings |
| `2017_Yellow_Taxi_Trip_Data.csv` | Raw NYC TLC Yellow Taxi trip dataset |
| `nyc_preds_means.csv` | Supplementary predicted-mean data merged in for Part 5 modeling |

---

## 🔬 Methodology

### Part 1 — Initial Data Exploration
- Loaded and inspected the raw trip dataset (`.info()`, `.describe()`, sorting by key variables).
- Identified data quality issues, including **negative fare values** and datetime-typed fields.
- Concluded that `trip_distance` and `total_amount` were the two variables most likely to be useful for a predictive fare model.

### Part 2 — Exploratory Data Analysis
- Visualized distributions of `trip_distance`, `total_amount`, and `tip_amount` using boxplots and histograms.
- Analyzed ride volume and revenue trends by **month, day of week, and drop-off location**.
- Investigated whether spatial clustering of drop-off locations reflected genuine ride patterns vs. sampling artifacts, using a simulated coordinate-distance experiment.
- Engineered a `trip_duration` field from pickup/drop-off timestamps.

### Part 3 — Hypothesis Testing
- Tested whether **payment type** (credit card vs. cash) is associated with a difference in average total fare amount.
- Ran a **two-sample t-test** at a 5% significance level.
- **Result:** p-value < 0.05 → rejected the null hypothesis. Customers paying by credit card show a statistically significant difference in average fare compared to cash payers.

### Part 4 — Regression Analysis
- Cleaned and imputed outliers in `trip_distance`, `fare_amount`, and `duration` using an IQR-based capping approach (`Q3 + 6×IQR`).
- Engineered key features: `mean_distance` and `mean_duration` (per pickup–dropoff pair), `day`, `month`, and a binary `rush_hour` flag.
- Assessed multicollinearity via a correlation matrix/heatmap — `mean_distance` and `mean_duration` were both highly correlated with the target (`fare_amount`) and with each other (r ≈ 0.87).
- Standardized features and trained a **multiple linear regression** model (80/20 train-test split).
- Evaluated using R², MAE, MSE, and RMSE, and inspected residual plots for bias/heteroscedasticity.

### Part 5 — Model Development
- Framed a new target: predicting **"generous" tippers** (tip ≥ 20% of pre-tip fare), restricted to credit-card payments (cash tips aren't reliably captured in the data).
- Engineered time-of-day bins (`am_rush`, `daytime`, `pm_rush`, `nighttime`), day/month features, and merged in supplementary mean-prediction data (`nyc_preds_means.csv`).
- Addressed a **moderately imbalanced target** (~1/3 of riders classified as generous) and selected **F1-score** as the primary metric, balancing the business cost of false positives (drivers expecting a tip that doesn't come) against false negatives.
- Benchmarked **Random Forest** and **XGBoost** classifiers using `GridSearchCV` for hyperparameter tuning.
- Reviewed feature importance and a confusion matrix to understand the model's error profile.
- Explicitly discussed the **ethical implications** of deploying a tip-prediction model to drivers (e.g. risk of behavioral bias, unequal service).

---

## 📊 Key Results

### Regression Model (Fare Prediction)
| Metric | Value |
|---|---|
| R² | 0.868 |
| MAE | 2.13 |
| MSE | 14.34 |
| RMSE | 3.80 |

- **86.8%** of the variance in fare amount is explained by the model.
- `mean_distance` was the strongest predictor: **fare increases by ~$2 per mile traveled** (or ~$7.13 per 3.57 miles).
- The model provides a reliable baseline fare estimate suitable for downstream product use, such as an in-app fare preview for riders.

### Classification Model (Generous Tipper Prediction)
- **XGBoost outperformed Random Forest** on F1-score by a modest margin (~0.04 higher), though both models had limited overall predictive strength given the available features.
- The model produced roughly twice as many **false negatives** as false positives — a favorable error profile for this use case, since it's better for a driver to be pleasantly surprised by an unexpected tip than to be let down by a predicted one that didn't materialize.
- **Ethical consideration:** a tip-prediction model risks introducing bias into driver behavior (e.g., avoiding riders predicted to tip poorly) and was flagged as a factor to weigh carefully before any deployment.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, Tableau
- **Statistics:** `scipy.stats`
- **Modelling:** `scikit-learn` (Linear Regression, Random Forest), `xgboost`
- **Model persistence:** `pickle`

---

## 📂 Datasets

- [`2017_Yellow_Taxi_Trip_Data.csv`](2017_Yellow_Taxi_Trip_Data.csv) — Raw NYC TLC Yellow Taxi trip-level data (2017)
- [`nyc_preds_means.csv`](nyc_preds_means.csv) — Supplementary mean-prediction data merged in for the Part 5 classification model

---

## 📑 Executive Summaries

- [Part 1 — Initial Data Exploration](Part_1_Initial_Data_Exploration_executive_summary.pptx)
- [Part 2 — Exploratory Data Analysis](Part_2_Exploratory_Data_Analysis_executive_summary.pptx)
- [Part 3 — Hypothesis Testing](Part_3_Hypothesis_Testing_executive_summary.pptx)
- [Part 4 — Regression Analysis](Part_4_Regression_Analysis_executive_summary.pptx)
- [Part 5 — Model Development](Part_5_Model_Development_executive_summary.pptx)

---

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/Automatidata_Google_Advanced_Data_Analysis.git
cd Automatidata_Google_Advanced_Data_Analysis
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
jupyter notebook
```

Open any of the five notebooks in order — each is self-contained but builds conceptually on the previous part. Update the data-loading file paths at the top of each notebook to point to the CSVs in this repository.

---

## ⚠️ Notes & Assumptions

- The Part 3 A/B test assumes riders were randomly and consistently assigned to a payment method for the purposes of the exercise — a simplification made for this educational project, since the real dataset does not reflect a true randomized experiment.
- The Part 5 tip-prediction analysis is restricted to credit-card transactions, since cash tips are not reliably recorded in the TLC dataset.
- This project was completed as part of a structured certificate curriculum (Google Advanced Data Analytics) using the PACE framework (**P**lan, **A**nalyze, **C**onstruct, **E**xecute) at each stage.

---

## 👤 Author

**Anurag Dash**

---

## 📄 License

Shared for educational and portfolio purposes.
