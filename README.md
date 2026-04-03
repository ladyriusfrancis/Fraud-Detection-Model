# sample transaction fraud detection model

i built this to show how i think about fraud detection end to end — not just the model, but the feature engineering, the SQL monitoring layer, the KPI framework, and how everything connects in a real production environment. this covers behavioral biometrics, device profiling, XGBoost with Optuna tuning, SHAP explainability, and Snowflake-ready SQL scripts.

---

## what it does

this is a full fraud detection workflow modeled after how fraud risk analytics teams actually operate at scale — working with platforms like Actimize, LexisNexis Risk Solutions, and behavioral intelligence tools like BioCatch.

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

## repo structure

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

## feature engineering

i grouped everything into five categories based on how production fraud pipelines actually feed data into models.

### core transaction features
transaction amount, amount-to-credit-limit ratio, channel (in-store, online, mobile, ATM, phone), merchant category code, account age, and prior dispute history.

### time features
cyclical hour encoding using sin/cos so the model doesn’t think 11pm and midnight are far apart. night transaction flag, weekend flag, holiday season flag.

### velo signals
this is where a lot of the detection power comes from. transaction count and dollar sum across 1h, 6h, 24h, and 7-day windows. distinct merchant and country counts over 24h. velocity acceleration ratios for catching 1h/24h surge patterns. amount z-score against the customer’s own 30-day baseline.

### behavioral biometrics *(BioCatch integration)*
session duration, typing speed, mouse linearity, copy-paste detection, login attempt counts. i also engineered a bot_signal feature (short session + copy-paste behavior) and a login_stress_score to flag unusual session patterns.

### device profiling *(LexisNexis ThreatMetrix integration)*
device age, known device flag, IP risk score, VPN/proxy/Tor detection, and geographic mismatch between billing country and IP country. engineered features include new_device_high_ip_risk and cnp_with_anonymizer (card-not-present + VPN). also incorporated a LexisNexis-style composite risk score combining identity, network, and velocity risk signals.

---

## SQL scripts

all written for Snowflake but easy to adapt for Redshift or SQL Server. these reflect the kind of queries you’d actually use for model monitoring in Actimize or a similar platform.

| Script | Purpose |
|--------|---------|
| `01_feature_extraction.sql` | full feature extraction pipeline using CTEs across 7 data sources |
| `02_velocity_checks.sql` | 6 real-time velocity fraud rules (card testing, impossible travel, spend spike) |
| `03_model_kpi_report.sql` | monthly KPI report: confusion matrix by score band, MoM trend, PSI drift |
| `04_false_positive_analysis.sql` | FP profiling by MCC, account age, threshold sensitivity at 9 cutoffs |

---

## model development

### baseline: logistic regression
started here because you need something interpretable to explain to stakeholders and validate that the feature set makes sense before going more complex.

### primary: XGBoost with Optuna Tuning
	∙	imbalance handling with scale_pos_weight (~38.5x for a 2.5% fraud rate)
	∙	optimized on PR-AUC instead of ROC-AUC (more on that below)
	∙	50-trial Optuna search using TPE sampler with 5-fold stratified CV
	∙	time-based train/test split — no random shuffle, because in production you’re always scoring future transactions

### why PR-AUC over ROC-AUC?
with a 2.5% fraud rate, a model that just predicts everything as legit still gets a ROC-AUC around 0.50. PR-AUC starts at 0.025 for that same dummy model, so it actually punishes you for missing fraud. it’s a better measure of how the model performs on the thing you care about — catching fraud without drowning analysts in false positives.

---

## kpi framework

these are the metrics i track to evaluate whether a model is actually production-ready, not just academically interesting.

| KPI | Definition | Target |
|-----|-----------|--------|
| Detection Rate | TP / (TP + FN) | ≥ 75% |
| Precision | TP / (TP + FP) | ≥ 60% |
| False Positive Rate | FP / (FP + TN) | ≤ 0.50% |
| FP per TP | FP / TP | ≤ 8x |
| Dollar Detection Rate | Fraud $ caught / Total fraud $ | ≥ 70% |
| PSI (monthly) | Score distribution stability | < 0.10 |

---

## this how you run it

### install dependecies
```bash
pip install -r requirements.txt
```

### 1. generate the data
```bash
cd data
python generate_synthetic_data.py
# Output: transactions.csv (100,000 rows)
```

### 2. run the notebook
open and run `notebooks/fraud_detection_analysis.ipynb` in Jupyter.

notebook covers:
open notebooks/fraud_detection_analysis.ipynb in Jupyter. it walks through everything: data quality checks, EDA, feature engineering, model training (LR + XGBoost), KPI reporting, threshold analysis, SHAP explainability, and business impact.

### 3. train from the command line
```bash
cd src
python model_training.py --data-path ../data/transactions.csv --n-trials 50 --threshold 0.50
```

---

## model validation

see [`reports/model_validation_report.md`](reports/model_validation_report.md) — covers SHAP feature importance, performance by score band, financial impact analysis, fairness assessment across customer segments and account age, PSI history, and a complete sign-off checklist.

---

## key design decisions

**time-based train/test split** — random shuffle on time-series fraud data causes data leakage. in production you’re always scoring transactions you haven’t seen yet, so the test set needs to reflect that..

**PR-AUC as the optimization target** — ROC-AUC looks good on paper for imbalanced data but doesn’t tell you much about how well you’re actually catching fraud. PR-AUC is what matters when you’re building something that has to work operationally.

**dual-threshold strategy** — one threshold doesn’t cut it. high confidence scores get auto-declined, medium scores go to analyst review. this reduces friction for customers on the clear-cut cases and focuses analyst time where it actually matters.

**behavioral + device features** — transaction data alone isn’t enough for modern fraud. behavioral biometrics and device intelligence features (ThreatMetrix-style) ended up being some of the top SHAP contributors, which validates why teams invest in those integrations.

---

## techs

- **python** — pandas, scikit-learn, XGBoost, SHAP, Optuna, matplotlib, seaborn
- **SQL** — Snowflake-compatible CTEs for feature extraction and KPI reporting
- **platforms referenced** — Actimize, LexisNexis Risk Solutions, LexisNexis ThreatMetrix, BioCatch

---

## disclaimer

this is all fake data. **synthetically generated data** for all my corporate jargon speakers. patterns are designed to mimic real fraud patterns, though. 
