# Transaction Fraud Detection Model

End-to-end fraud detection system for banking transaction data — covering feature engineering with behavioral biometrics and device profiling, XGBoost model development, KPI reporting, and SHAP explainability.

---

## Overview

This repository demonstrates a complete fraud model development and evaluation workflow aligned with financial services best practices. It reflects the processes used by fraud risk analytics teams working with platforms like **Actimize**, **LexisNexis Risk Solutions**, and behavioral intelligence providers.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.961 |
| PR-AUC | 0.870 |
| Detection Rate | 82.4% |
| False Positive Rate | 0.31% |
| Model Type | XGBoost (v2.3) |
| Dataset | 100,000 synthetic transactions |
| Fraud Rate | ~2.5% |

---

## Repository Structure

```
fraud-detection-model/
│
├── data/
│   └── generate_synthetic_data.py    # Synthetic transaction data generator
│
├── sql/
│   ├── 01_feature_extraction.sql     # Feature engineering via SQL (CTE-based)
│   ├── 02_velocity_checks.sql        # Fraud velocity rules (6 rule types)
│   ├── 03_model_kpi_report.sql       # Monthly KPI reporting (detection rate, FPR, PSI)
│   └── 04_false_positive_analysis.sql # FP profiling and threshold sensitivity
│
├── src/
│   ├── feature_engineering.py        # Python feature pipeline (sklearn-compatible)
│   ├── model_training.py             # Model training + Optuna tuning
│   └── evaluation.py                 # KPI plotting, SHAP, threshold analysis
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb  # End-to-end analysis (EDA → model → KPIs → SHAP)
│
├── reports/
│   └── model_validation_report.md    # Formal model validation documentation
│
├── requirements.txt
└── .gitignore
```

---

## Feature Engineering

Features are grouped into five categories reflecting real production data pipelines:

### Core Transaction Features
- Transaction amount, amount-to-credit-limit ratio
- Channel (in-store, online, mobile, ATM, phone)
- Merchant category code (MCC)
- Account age and prior dispute history

### Time Features
- Cyclical hour encoding (sin/cos) to handle 23h→0h boundary
- Night transaction flag, weekend flag, holiday season flag

### Velocity Signals
- Transaction count and dollar sum: 1h, 6h, 24h, 7-day windows
- Distinct merchant count and country count (24h)
- Velocity acceleration ratios (1h/24h surge detection)
- Amount z-score vs. 30-day account baseline

### Behavioral Biometrics *(BioCatch integration)*
- Session duration, typing speed (WPM), mouse linearity score
- Copy-paste event detection, login attempt count
- Engineered: `bot_signal` (short session + copy-paste)
- Engineered: `login_stress_score`

### Device Profiling *(LexisNexis ThreatMetrix integration)*
- Device age (days since first seen), known device flag
- IP risk score (0–100), VPN/proxy detected, Tor exit node flag
- Geographic IP mismatch (billing country ≠ IP country)
- Engineered: `new_device_high_ip_risk` (new device + high IP risk)
- Engineered: `cnp_with_anonymizer` (card-not-present + VPN)
- LexisNexis composite risk score (identity + network + velocity risk)

---

## SQL Scripts

The SQL scripts are written for **Snowflake** (compatible with Redshift and SQL Server with minor dialect changes). They reflect the kinds of queries used for model KPI monitoring in Actimize or similar fraud platforms.

| Script | Purpose |
|--------|---------|
| `01_feature_extraction.sql` | Full feature extraction pipeline using CTEs across 7 data sources |
| `02_velocity_checks.sql` | 6 real-time velocity fraud rules (card testing, impossible travel, spend spike) |
| `03_model_kpi_report.sql` | Monthly KPI report: confusion matrix by score band, MoM trend, PSI drift |
| `04_false_positive_analysis.sql` | FP profiling by MCC, account age, threshold sensitivity at 9 cutoffs |

---

## Model Development

### Baseline: Logistic Regression
Provides interpretable coefficients for stakeholder communication and model logic review.

### Primary: XGBoost with Optuna Tuning
- **Imbalance handling:** `scale_pos_weight` (~38.5x for 2.5% fraud rate)
- **Optimization target:** PR-AUC (more appropriate than ROC-AUC for imbalanced fraud data)
- **Hyperparameter search:** Optuna TPE sampler, 50 trials, 5-fold stratified CV
- **Validation approach:** Time-based train/test split (no random shuffle) to prevent data leakage

### Why PR-AUC over ROC-AUC?
With a 2.5% fraud rate, a model predicting all transactions as legitimate achieves ROC-AUC ≈ 0.50 but PR-AUC ≈ 0.025. PR-AUC better reflects performance on the minority fraud class and is more meaningful for operational decision-making.

---

## KPI Framework

| KPI | Definition | Target |
|-----|-----------|--------|
| Detection Rate | TP / (TP + FN) | ≥ 75% |
| Precision | TP / (TP + FP) | ≥ 60% |
| False Positive Rate | FP / (FP + TN) | ≤ 0.50% |
| FP per TP | FP / TP | ≤ 8x |
| Dollar Detection Rate | Fraud $ caught / Total fraud $ | ≥ 70% |
| PSI (monthly) | Score distribution stability | < 0.10 |

---

## Setup & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Generate Synthetic Data
```bash
cd data
python generate_synthetic_data.py
# Output: transactions.csv (100,000 rows)
```

### 2. Run the Full Analysis
Open and run `notebooks/fraud_detection_analysis.ipynb` in Jupyter.

The notebook covers:
1. Data quality checks
2. Exploratory data analysis
3. Feature engineering
4. Model training (LR baseline + XGBoost)
5. KPI reporting and confusion matrix
6. Threshold sensitivity analysis
7. SHAP explainability (summary, bar, waterfall)
8. Business impact quantification

### 3. Train Models via CLI
```bash
cd src
python model_training.py --data-path ../data/transactions.csv --n-trials 50 --threshold 0.50
```

---

## Model Validation

See [`reports/model_validation_report.md`](reports/model_validation_report.md) for the complete model validation documentation, including:
- Feature importance rankings (SHAP)
- Performance by score band
- Financial impact analysis
- Fairness assessment (by customer segment and account age)
- Population Stability Index (PSI) history
- Full sign-off checklist

---

## Key Design Decisions

**Time-based train/test split** — Prevents data leakage and simulates real deployment where models score future transactions. Never use random shuffle for time-series fraud data.

**PR-AUC as optimization target** — ROC-AUC is misleading for highly imbalanced fraud data. PR-AUC directly measures precision-recall trade-off on the minority class.

**Dual-threshold strategy** — A single threshold conflates auto-decline (high confidence) with analyst review (medium confidence). Separating these reduces unnecessary customer friction for the highest-confidence cases.

**Behavioral + device features** — Raw transaction features alone are insufficient for modern fraud. Behavioral biometrics and device intelligence (LexisNexis ThreatMetrix) are among the top SHAP contributors, validating the integration investment.

---

## Technologies

- **Python** — pandas, scikit-learn, XGBoost, SHAP, Optuna, matplotlib, seaborn
- **SQL** — Snowflake-compatible CTEs for feature extraction and KPI reporting
- **Platforms referenced** — Actimize, LexisNexis Risk Solutions, LexisNexis ThreatMetrix, BioCatch

---

## Disclaimer

All data in this repository is **synthetically generated**. No real customer data, account information, or proprietary bank data is included. Synthetic patterns are designed to reflect realistic fraud distributions for modeling demonstration purposes only.
