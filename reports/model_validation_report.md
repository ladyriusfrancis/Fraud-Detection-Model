# Fraud Detection Model — Validation Report

**Model Name:** Transaction Fraud Detection Model
**Model Version:** v2.3
**Model Type:** XGBoost Gradient Boosted Trees
**Validation Date:** 2024-Q4
**Author:** Fraud Model Analytics
**Status:** ✅ Approved for Production

---

## 1. Executive Summary

The v2.3 Transaction Fraud Detection Model achieves a **PR-AUC of 0.87** and a **detection rate of 82.4%** at the 0.50 decision threshold, with a false positive rate of 0.31%. Compared to the prior model version (v2.1), this represents:

- **+6.2 pp** improvement in detection rate
- **−0.09 pp** reduction in false positive rate
- **+$148,000** estimated additional monthly fraud loss prevented

The model is recommended for production deployment. A dual-threshold strategy is recommended (auto-decline ≥ 0.80; review queue 0.50–0.79) to optimize analyst workload.

---

## 2. Model Overview

### 2.1 Purpose and Scope

This model identifies potentially fraudulent transactions in real time across all card transaction channels: in-store, online, mobile, ATM, and phone. The model produces a continuous score (0–1000, normalized) at transaction time, with higher scores indicating greater fraud probability.

### 2.2 Model Architecture

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost (gradient boosted trees) |
| n_estimators | 450 |
| max_depth | 5 |
| learning_rate | 0.05 |
| subsample | 0.80 |
| colsample_bytree | 0.75 |
| scale_pos_weight | ~38.5x (class imbalance adjustment) |
| Optimization target | PR-AUC (average precision) |
| Hyperparameter tuning | Optuna TPE (50 trials, 5-fold CV) |

### 2.3 Data Sources

| Source | Type | Refresh Rate | Notes |
|--------|------|-------------|-------|
| Core banking transaction warehouse | Structured | Real-time | Primary transaction attributes |
| BioCatch behavioral biometrics | API feed | Per-session | Typing, mouse, session signals |
| LexisNexis ThreatMetrix | API feed | Per-transaction | Device intelligence, IP risk |
| LexisNexis Risk Solutions | Batch API | Daily | Composite identity risk scores |
| Fraud case management (chargebacks) | Structured | T+30 to T+90 | Ground truth labels |

### 2.4 Decision Threshold Policy

| Band | Score Range | Action |
|------|------------|--------|
| Auto-Decline | ≥ 800 | Transaction declined automatically |
| Review Queue | 500–799 | Routed to fraud analyst review |
| Monitor | 300–499 | Flagged for passive monitoring |
| Pass | < 300 | Transaction approved |

---

## 3. Feature Documentation

### 3.1 Feature Groups and Importance

| Rank | Feature | SHAP Importance | Category |
|------|---------|----------------|----------|
| 1 | `ln_risk_score_norm` | 0.142 | LexisNexis |
| 2 | `ip_risk_score_norm` | 0.118 | Device profiling |
| 3 | `amount_to_limit_ratio` | 0.097 | Core transaction |
| 4 | `amount_zscore` | 0.089 | Amount deviation |
| 5 | `txn_count_1h` | 0.081 | Velocity |
| 6 | `bot_signal` | 0.074 | Behavioral biometrics |
| 7 | `device_age_days` | 0.068 | Device profiling |
| 8 | `new_device_high_ip_risk` | 0.061 | Device profiling |
| 9 | `session_duration_sec` | 0.054 | Behavioral biometrics |
| 10 | `velocity_accel_1h_24h` | 0.049 | Velocity ratio |

*Note: SHAP values are mean absolute contributions. Full feature importances in `notebooks/fraud_detection_analysis.ipynb`, Section 6.*

### 3.2 Key Engineered Features

**`bot_signal`** — Binary flag (1 if `session_duration_sec < 30` AND `copy_paste_detected = 1`). Identifies automated credential stuffing and bot-driven account takeover.

**`new_device_high_ip_risk`** — Binary flag (1 if `device_age_days < 7` AND `ip_risk_score > 60`). Strong combined signal for account takeover using compromised credentials on a new device.

**`velocity_accel_1h_24h`** — Ratio of 1-hour transaction count to 24-hour count. Values near 1.0 indicate sudden bursts inconsistent with typical account behavior.

**`amount_zscore`** — Standard score of current transaction amount relative to account's 30-day distribution. Values >3 indicate statistically anomalous amounts.

---

## 4. Model Performance

### 4.1 Overall KPIs (Test Set, Threshold = 0.50)

| KPI | Value | Benchmark | Status |
|-----|-------|-----------|--------|
| ROC-AUC | 0.9614 | ≥ 0.93 | ✅ Pass |
| PR-AUC | 0.8702 | ≥ 0.80 | ✅ Pass |
| Detection Rate | 82.4% | ≥ 75% | ✅ Pass |
| Precision | 71.3% | ≥ 60% | ✅ Pass |
| False Positive Rate | 0.31% | ≤ 0.50% | ✅ Pass |
| FP per TP | 5.2x | ≤ 8x | ✅ Pass |

### 4.2 Performance by Score Band

| Score Band | Transactions | Fraud | Detection Rate | Precision | FPR |
|-----------|-------------|-------|---------------|-----------|-----|
| 800–1000 (High) | 412 | 318 | 41.2% | 77.2% | — |
| 600–799 (Med-High) | 1,847 | 918 | 41.2% | 49.7% | — |
| 400–599 (Medium) | 3,219 | 227 | — | — | — |
| 0–399 (Low) | 14,522 | 49 | — | — | — |

*Bands are operational tiers. Detection rate is cumulative from highest band.*

### 4.3 Financial Impact

| Metric | Value |
|--------|-------|
| Total fraud exposure (test set) | $2,847,391 |
| Fraud caught by model | $2,345,410 (82.4%) |
| Fraud escaped model | $501,981 (17.6%) |
| Estimated loss prevented (85% recovery) | $1,993,599 |
| FP review cost ($10/alert) | $57,200 |
| Net financial benefit | $1,936,399 |

---

## 5. Fairness and Bias Assessment

The model was evaluated for disparate impact across customer segments:

| Segment | Detection Rate | FPR | FP Burden |
|---------|---------------|-----|-----------|
| Mass Market | 81.9% | 0.33% | Baseline |
| Mass Affluent | 83.1% | 0.28% | −15% |
| High Net Worth | 82.7% | 0.24% | −27% |
| New Accounts (<90d) | 84.2% | 0.89% | +170% |

**Finding:** New accounts show elevated FPR (0.89% vs 0.31% overall). This is expected — new accounts lack behavioral baseline. Recommend suppression rule: do not auto-decline accounts <30 days old; route to review queue instead.

---

## 6. Model Stability Assessment (Population Stability Index)

PSI is computed monthly comparing current score distributions to the development baseline.

| Month | PSI | Status |
|-------|-----|--------|
| 2024-07 | 0.021 | ✅ Stable |
| 2024-08 | 0.018 | ✅ Stable |
| 2024-09 | 0.033 | ✅ Stable |
| 2024-10 | 0.041 | ✅ Stable |
| 2024-11 | 0.078 | ✅ Stable |

PSI thresholds: < 0.10 = Stable | 0.10–0.25 = Monitor | > 0.25 = Retrain

---

## 7. Limitations and Risk Factors

**Label lag:** Fraud labels are sourced from chargeback/dispute resolution, which can take 30–90 days. The model's training data represents historical fraud patterns, which may lag emerging attack vectors.

**Synthetic identity fraud:** The model has limited coverage of synthetic identity fraud (SIF), as SIF often manifests at account opening rather than transaction time. Recommend supplementing with an application-level model.

**Behavioral biometric coverage:** Session biometric features (BioCatch) are unavailable for 18% of transactions (ATM and phone channels). Missing values are imputed to population median, which may slightly reduce performance for these channels.

**Third-party score availability:** LexisNexis composite scores are batch-refreshed daily. For real-time transactions, the score reflects the prior day's data, introducing up to 24 hours of staleness.

---

## 8. Validation Testing Checklist

| Test | Result |
|------|--------|
| Data quality validation | ✅ Pass — 0 missing values in training set |
| Temporal holdout validation | ✅ Pass — time-based split used throughout |
| Cross-validation stability (5-fold) | ✅ Pass — CV std < 0.01 |
| Performance on OOT sample | ✅ Pass — PR-AUC 0.862 on 6-month OOT |
| Feature leakage check | ✅ Pass — no post-event features included |
| Class balance validation | ✅ Pass — scale_pos_weight applied |
| Fairness analysis | ✅ Pass — new account suppression rule added |
| PSI monitoring | ✅ Stable — PSI < 0.10 for all production months |
| SHAP logic review | ✅ Pass — top features align with domain expectations |
| IT security review | ✅ Pass — no PII in model features |

---

## 9. Approval and Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Model Developer | Fraud Model Analytics | 2024-Q4 | ✅ |
| Model Owner | Senior Fraud Model Analyst | 2024-Q4 | ✅ |
| Risk Management | Model Risk Management | 2024-Q4 | ✅ |
| Compliance | BSA/AML Compliance | 2024-Q4 | ✅ |

---

## 10. Change Log

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2022-Q1 | Initial logistic regression baseline |
| v1.5 | 2022-Q4 | Migrated to XGBoost; added velocity features |
| v2.0 | 2023-Q3 | Integrated LexisNexis ThreatMetrix device signals |
| v2.1 | 2024-Q1 | Added behavioral biometrics (BioCatch integration) |
| v2.2 | 2024-Q2 | Optuna hyperparameter tuning; added bot_signal feature |
| v2.3 | 2024-Q4 | Added new_device_high_ip_risk; new account suppression rule; retrained on 18 months of data |
