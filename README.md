# 🌊 Explainable Machine Learning for Sustainable Urban Lake Monitoring

## Proxy-Based Harmful Algal Bloom Risk Assessment and Early Warning

This project presents an explainable Machine Learning and Decision Support System (DSS) for assessing **proxy-derived Harmful Algal Bloom (HAB) risk** in urban water bodies using routinely collected water-quality data from the **Karnataka State Pollution Control Board (KSPCB)**.

The framework combines proxy-risk assessment, machine-learning-based next-month early warning, explainable AI (SHAP), exploratory time-series forecasting, and an interactive Streamlit dashboard to support sustainable urban lake monitoring.

> **Important:** The system estimates proxy-derived HAB risk from physicochemical water-quality conditions. It does not detect or confirm actual harmful algal blooms. Field sampling and biological/laboratory verification are required for HAB confirmation.

---

## 🎯 Objectives

The main objectives of the project are to:

- Construct a transparent proxy HAB Risk Score from routinely monitored water-quality parameters.
- Predict next-month **Lower Proxy Risk** or **Elevated Proxy Risk** conditions.
- Compare Logistic Regression, Random Forest, and XGBoost models.
- Evaluate temporal and spatial generalization of the models.
- Explain model predictions using SHAP.
- Analyse model robustness using threshold sensitivity and feature-group ablation.
- Explore short-term proxy-risk forecasting using time-series models.
- Develop an interactive Decision Support System for monitoring prioritization.

---

## 📊 Dataset

The study uses monthly water-quality observations obtained from **KSPCB reports**.

### Dataset Summary

| Attribute | Description |
|---|---|
| Period | July 2023 – November 2025 |
| Reporting Months | 27 |
| Total Observations | 2,700 |
| Raw Station Codes | 151 |
| Stations in Temporal ML Dataset | 144 |
| Valid Consecutive-Month Pairs | 1,925 |

### Water-Quality Parameters

Eight physicochemical parameters are used:

1. Dissolved Oxygen (DO)
2. pH
3. Biochemical Oxygen Demand (BOD)
4. Chemical Oxygen Demand (COD)
5. Nitrate
6. Ammonical Nitrogen
7. Phosphate
8. Turbidity

Missing values are handled using station-specific median imputation followed by the overall median where required.

---

## 🧪 Proxy HAB Risk Assessment

Since consistent biological HAB indicators such as chlorophyll-a, cyanobacterial abundance, phycocyanin, and algal toxins are unavailable, the project constructs a **proxy HAB Risk Score** from routinely available physicochemical measurements.

The parameters are organized into three conceptual risk groups:

### Nutrient Risk
- Nitrate
- Ammonical Nitrogen
- Phosphate

### Organic-Pollution Risk
- BOD
- COD

### Ecological Risk
- Dissolved Oxygen
- pH
- Turbidity

The final proxy-risk score ranges from **0 to 100**.

### Risk Categories

| Proxy Risk Score | Risk Category |
|---|---|
| ≤ 34.34 | Low |
| 34.34 – 46.40 | Moderate |
| 46.40 – 66.80 | High |
| > 66.80 | Very High |

These thresholds are **dataset-relative quartiles** and should not be interpreted as regulatory HAB thresholds.

---

## 🤖 Machine Learning

The system performs leakage-aware next-month prediction.

Water-quality measurements from month **t** are used to predict the independently calculated proxy-risk condition at month **t+1**.

For operational early warning, the four risk classes are consolidated into:

- **Lower Proxy Risk:** Low + Moderate
- **Elevated Proxy Risk:** High + Very High

### Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

---

## 📈 Model Evaluation

Three complementary validation strategies are used:

### 1. Chronological Holdout
Earlier observations are used for training and later observations for testing.

### 2. Expanding Walk-Forward Validation
Evaluates model performance across time while preserving temporal order.

### 3. Station-Held-Out Validation
Evaluates generalization to monitoring stations that were not included during model training.

### Chronological Holdout Results

| Model | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6499 | 0.5335 | 0.5956 | 0.6818 |
| Random Forest | 0.6403 | 0.5757 | 0.6073 | **0.6864** |
| XGBoost | 0.6343 | **0.6079** | **0.6164** | 0.6732 |

XGBoost is retained for operational screening because it provides stronger **Elevated Proxy Risk recall and F1-score**, which are important for early-warning applications.

This does not imply that XGBoost is universally superior to the other evaluated models.

---

## 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to interpret XGBoost predictions.

The most influential features based on mean absolute SHAP values are:

| Rank | Feature | Mean \|SHAP\| |
|---|---|---:|
| 1 | Phosphate | 0.4610 |
| 2 | COD | 0.4526 |
| 3 | Nitrate | 0.4064 |
| 4 | Ammonical Nitrogen | 0.3664 |
| 5 | Turbidity | 0.2886 |
| 6 | Dissolved Oxygen | 0.1888 |
| 7 | BOD | 0.1859 |
| 8 | pH | 0.1818 |

SHAP values represent **model attribution**, not causal relationships between water-quality variables and HAB occurrence.

---

## 🧪 Robustness Analysis

The framework also includes:

- Threshold sensitivity analysis
- Feature-group ablation
- Probability calibration analysis
- Temporal validation
- Spatial/station-held-out validation

Alternative binary cutoffs showed high agreement with the original dataset-relative threshold, helping evaluate the robustness of the screening framework.

---

## ⏳ Time-Series Forecasting

Short-term proxy-risk forecasting is explored for **Rayasandra Lake (STN 4531)**.

The following approaches are compared:

- Naive Persistence
- ARIMA
- Experimental LSTM

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Persistence | **4.282** | 5.697 |
| ARIMA (1,0,1) | 4.730 | **5.405** |
| LSTM | 4.358 | 5.555 |

ARIMA is retained for the DSS forecasting demonstration because of its RMSE performance and interpretability for the short time series.

The forecasting component should be considered **exploratory** because of the limited number of available monthly observations.

---

## 💻 Decision Support System

An interactive DSS is developed using **Streamlit**.

### Main Functions

The dashboard provides:

- Current proxy HAB Risk Score
- Risk classification
- Historical risk visualization
- Next-month ML-based screening
- Model confidence information
- SHAP-based explanations
- Exploratory forecasting
- Monitoring recommendations

### Monitoring Recommendations

| Risk Level | Recommended Action |
|---|---|
| Low | Routine monitoring |
| Moderate | Enhanced monitoring |
| High | Field investigation and sampling |
| Very High | Priority sampling and laboratory verification |

These recommendations are intended as **decision-support guidance**, not automated regulatory decisions.

---

## 🛠️ Technology Stack

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Explainable AI
- SHAP

### Time-Series Analysis
- ARIMA
- LSTM

### Visualization
- Matplotlib
- Plotly

### Application
- Streamlit

---



## 🌱 Sustainability Relevance

The framework explores how existing regulatory water-quality observations can be reused for **resource-aware monitoring prioritization**.

Instead of claiming reductions in energy, carbon emissions, cost, or field workload, the system is designed to help identify locations and periods that may warrant additional monitoring and verification.

---

## ⚠️ Limitations

- The prediction target is proxy-derived rather than based on biologically confirmed HAB observations.
- Direct biological indicators such as chlorophyll-a and cyanobacterial abundance are unavailable.
- Monthly observations may miss short-duration bloom dynamics.
- Meteorological and hydrological variables are not included.
- The risk thresholds are dataset-specific.
- XGBoost confidence scores should not be interpreted as calibrated probabilities of actual HAB occurrence.
- Forecasting is based on a limited monthly time series.
- External validation using biologically confirmed HAB observations is required.

---

## 🔮 Future Work

Future extensions can include:

- Direct chlorophyll-a and cyanobacterial measurements
- Phycocyanin and algal toxin measurements
- IoT-based real-time water-quality sensors
- Meteorological and hydrological variables
- Satellite remote-sensing observations
- Higher-frequency monitoring
- Improved probability calibration
- External validation across additional lakes and cities
- Evaluation of real-world monitoring-efficiency benefits

---
