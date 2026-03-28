-- =============================================================================
-- 03_model_kpi_report.sql
-- =============================================================================
-- Purpose : Monthly model performance KPI report for fraud detection models.
--           Tracks key metrics required by model owners, risk management,
--           and internal audit.
--
-- KPIs tracked:
--   - Detection Rate (Recall) by score band
--   - False Positive Rate (FPR) and False Positive Ratio
--   - Fraud Loss Detected vs. Escaped
--   - Fraud Loss Prevented (estimated recovery rate applied)
--   - Alert Volume and Review Queue Metrics
--   - Model Stability (Population Stability Index proxy)
--
-- Author  : Fraud Model Analytics
-- Frequency: Monthly / ad-hoc
-- =============================================================================

-- Parameterize report period
-- :report_month  = 'YYYY-MM'   e.g., '2024-11'
-- :model_version = 'v2.3'


-- =============================================================================
-- PART 1 : Confusion Matrix by Score Band
-- =============================================================================

WITH scored_txns AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.transaction_date,
        t.amount,
        s.model_score,
        s.model_version,
        -- Score decile and band
        NTILE(10) OVER (ORDER BY s.model_score DESC)  AS score_decile,
        CASE
            WHEN s.model_score >= 900 THEN 'Band 1: 900-1000 (Highest Risk)'
            WHEN s.model_score >= 700 THEN 'Band 2: 700-899'
            WHEN s.model_score >= 500 THEN 'Band 3: 500-699'
            WHEN s.model_score >= 300 THEN 'Band 4: 300-499'
            ELSE                           'Band 5: 0-299 (Lowest Risk)'
        END                                            AS score_band,
        -- Decision threshold (review queue cutoff)
        CASE WHEN s.model_score >= 600 THEN 1 ELSE 0 END AS model_flagged,
        -- Ground truth (confirmed fraud from dispute/chargeback resolution)
        COALESCE(f.is_confirmed_fraud, 0)              AS is_actual_fraud,
        f.fraud_type,
        f.chargeback_amount
    FROM fraud_db.transactions              t
    JOIN fraud_db.model_scores              s
      ON t.transaction_id = s.transaction_id
     AND s.model_version = :model_version
    LEFT JOIN fraud_db.confirmed_fraud_labels f
      ON t.transaction_id = f.transaction_id
    WHERE DATE_TRUNC('month', t.transaction_date) = :report_month
),

confusion_by_band AS (
    SELECT
        score_band,
        COUNT(*)                                                          AS total_transactions,
        SUM(is_actual_fraud)                                              AS actual_fraud_count,
        SUM(CASE WHEN model_flagged = 1 AND is_actual_fraud = 1
                 THEN 1 ELSE 0 END)                                       AS true_positives,
        SUM(CASE WHEN model_flagged = 1 AND is_actual_fraud = 0
                 THEN 1 ELSE 0 END)                                       AS false_positives,
        SUM(CASE WHEN model_flagged = 0 AND is_actual_fraud = 0
                 THEN 1 ELSE 0 END)                                       AS true_negatives,
        SUM(CASE WHEN model_flagged = 0 AND is_actual_fraud = 1
                 THEN 1 ELSE 0 END)                                       AS false_negatives,
        SUM(CASE WHEN is_actual_fraud = 1 THEN amount ELSE 0 END)        AS fraud_amount_in_band,
        SUM(CASE WHEN model_flagged = 1 AND is_actual_fraud = 1
                 THEN amount ELSE 0 END)                                  AS fraud_amount_caught,
        SUM(CASE WHEN model_flagged = 0 AND is_actual_fraud = 1
                 THEN amount ELSE 0 END)                                  AS fraud_amount_escaped
    FROM scored_txns
    GROUP BY score_band
)

SELECT
    score_band,
    total_transactions,
    actual_fraud_count,
    true_positives,
    false_positives,
    true_negatives,
    false_negatives,

    -- Detection Rate (Recall) = TP / (TP + FN)
    ROUND(true_positives::FLOAT
          / NULLIF(true_positives + false_negatives, 0) * 100, 2)        AS detection_rate_pct,

    -- Precision = TP / (TP + FP)
    ROUND(true_positives::FLOAT
          / NULLIF(true_positives + false_positives, 0) * 100, 2)        AS precision_pct,

    -- False Positive Rate = FP / (FP + TN)
    ROUND(false_positives::FLOAT
          / NULLIF(false_positives + true_negatives, 0) * 100, 4)        AS false_positive_rate_pct,

    -- False Positive Ratio (per 1 TP, how many FPs reviewed?) = FP / TP
    ROUND(false_positives::FLOAT / NULLIF(true_positives, 0), 1)         AS fp_to_tp_ratio,

    -- Dollar-based detection
    ROUND(fraud_amount_in_band, 2)                                        AS total_fraud_exposure_usd,
    ROUND(fraud_amount_caught, 2)                                         AS fraud_detected_usd,
    ROUND(fraud_amount_escaped, 2)                                        AS fraud_escaped_usd,
    ROUND(fraud_amount_caught
          / NULLIF(fraud_amount_in_band, 0) * 100, 2)                    AS dollar_detection_rate_pct

FROM confusion_by_band
ORDER BY score_band;


-- =============================================================================
-- PART 2 : Overall Monthly KPI Summary
-- =============================================================================

SELECT
    :report_month                                                          AS report_month,
    :model_version                                                         AS model_version,

    -- Volume
    COUNT(*)                                                               AS total_transactions,
    SUM(is_actual_fraud)                                                   AS total_confirmed_fraud,
    ROUND(SUM(is_actual_fraud)::FLOAT / COUNT(*) * 100, 4)                AS fraud_rate_pct,

    -- Detection KPIs
    SUM(CASE WHEN model_flagged=1 AND is_actual_fraud=1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN model_flagged=1 AND is_actual_fraud=0 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN model_flagged=0 AND is_actual_fraud=1 THEN 1 ELSE 0 END) AS false_negatives,

    ROUND(SUM(CASE WHEN model_flagged=1 AND is_actual_fraud=1 THEN 1 ELSE 0 END)::FLOAT
          / NULLIF(SUM(is_actual_fraud), 0) * 100, 2)                     AS overall_detection_rate_pct,

    ROUND(SUM(CASE WHEN model_flagged=1 AND is_actual_fraud=0 THEN 1 ELSE 0 END)::FLOAT
          / NULLIF(SUM(CASE WHEN model_flagged=1 THEN 1 ELSE 0 END), 0) * 100, 2)
                                                                           AS false_positive_ratio_pct,

    -- Financial impact
    ROUND(SUM(CASE WHEN is_actual_fraud=1 THEN amount ELSE 0 END), 2)     AS total_fraud_exposure_usd,
    ROUND(SUM(CASE WHEN model_flagged=1 AND is_actual_fraud=1
                   THEN amount ELSE 0 END), 2)                            AS fraud_detected_usd,
    ROUND(SUM(CASE WHEN model_flagged=0 AND is_actual_fraud=1
                   THEN amount ELSE 0 END), 2)                            AS fraud_escaped_usd,

    -- Alert queue
    SUM(model_flagged)                                                     AS total_alerts_generated,
    ROUND(SUM(model_flagged)::FLOAT / COUNT(*) * 100, 3)                  AS alert_rate_pct

FROM scored_txns;


-- =============================================================================
-- PART 3 : Month-over-Month Trend (12-month rolling)
-- =============================================================================

SELECT
    DATE_TRUNC('month', t.transaction_date)::DATE                        AS month,
    s.model_version,
    COUNT(*)                                                              AS total_txns,
    SUM(COALESCE(f.is_confirmed_fraud, 0))                               AS fraud_count,
    ROUND(SUM(COALESCE(f.is_confirmed_fraud, 0))::FLOAT / COUNT(*) * 100, 4) AS fraud_rate_pct,
    -- Detection rate
    ROUND(
        SUM(CASE WHEN s.model_score >= 600
                  AND COALESCE(f.is_confirmed_fraud, 0) = 1
                 THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(SUM(COALESCE(f.is_confirmed_fraud, 0)), 0) * 100,
    2)                                                                    AS detection_rate_pct,
    -- FPR
    ROUND(
        SUM(CASE WHEN s.model_score >= 600
                  AND COALESCE(f.is_confirmed_fraud, 0) = 0
                 THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*) - SUM(COALESCE(f.is_confirmed_fraud, 0)), 0) * 100,
    4)                                                                    AS false_positive_rate_pct,
    -- Dollar detection
    ROUND(SUM(CASE WHEN s.model_score >= 600
                    AND COALESCE(f.is_confirmed_fraud, 0) = 1
                   THEN t.amount ELSE 0 END), 2)                         AS fraud_dollars_detected,
    ROUND(SUM(CASE WHEN s.model_score <  600
                    AND COALESCE(f.is_confirmed_fraud, 0) = 1
                   THEN t.amount ELSE 0 END), 2)                         AS fraud_dollars_escaped
FROM fraud_db.transactions          t
JOIN fraud_db.model_scores          s ON t.transaction_id = s.transaction_id
LEFT JOIN fraud_db.confirmed_fraud_labels f ON t.transaction_id = f.transaction_id
WHERE t.transaction_date >= DATEADD('month', -12, CURRENT_DATE)
GROUP BY DATE_TRUNC('month', t.transaction_date)::DATE, s.model_version
ORDER BY month, model_version;


-- =============================================================================
-- PART 4 : Population Stability Index (PSI) — Model Drift Monitoring
-- Compares current month score distribution vs. development sample baseline
-- PSI < 0.10: Stable | 0.10-0.25: Minor shift | > 0.25: Significant drift
-- =============================================================================

WITH current_dist AS (
    SELECT
        score_decile,
        COUNT(*)                              AS current_count,
        COUNT(*) / SUM(COUNT(*)) OVER ()      AS current_pct
    FROM scored_txns
    GROUP BY score_decile
),

baseline_dist AS (
    SELECT
        score_decile,
        COUNT(*)                              AS baseline_count,
        COUNT(*) / SUM(COUNT(*)) OVER ()      AS baseline_pct
    FROM (
        SELECT NTILE(10) OVER (ORDER BY s.model_score DESC) AS score_decile
        FROM fraud_db.model_scores s
        JOIN fraud_db.transactions t ON s.transaction_id = t.transaction_id
        WHERE s.model_version = :model_version
          AND DATE_TRUNC('month', t.transaction_date) = '2024-01'  -- development baseline month
    ) baseline_scored
    GROUP BY score_decile
)

SELECT
    c.score_decile,
    c.current_count,
    ROUND(c.current_pct * 100, 2)              AS current_pct,
    b.baseline_count,
    ROUND(b.baseline_pct * 100, 2)             AS baseline_pct,
    -- PSI contribution: (Actual% - Expected%) * ln(Actual% / Expected%)
    ROUND((c.current_pct - b.baseline_pct)
          * LN(c.current_pct / NULLIF(b.baseline_pct, 0)), 6)  AS psi_contribution
FROM current_dist  c
JOIN baseline_dist b ON c.score_decile = b.score_decile
ORDER BY c.score_decile;
