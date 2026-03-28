-- =============================================================================
-- 04_false_positive_analysis.sql
-- =============================================================================
-- Purpose : Deep-dive analysis of false positive alerts to identify reduction
--           opportunities without sacrificing fraud detection coverage.
--
--           False positive reduction is critical because:
--           - Each FP alert costs ~$8–15 in analyst review time
--           - FP-heavy models damage customer experience (declined good txns)
--           - Operational teams prioritize queues by FP burden
--
--           Findings from this analysis feed directly into threshold tuning,
--           rule overrides, and model retraining decisions.
--
-- Author  : Fraud Model Analytics
-- =============================================================================


-- =============================================================================
-- PART 1 : FP Profile by Merchant Category (where are FPs concentrated?)
-- =============================================================================

WITH alerts AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.amount,
        t.merchant_category_code    AS mcc,
        t.channel,
        t.transaction_date,
        s.model_score,
        COALESCE(f.is_confirmed_fraud, 0)  AS is_actual_fraud,
        CASE WHEN s.model_score >= 600
              AND COALESCE(f.is_confirmed_fraud, 0) = 0
             THEN 1 ELSE 0 END             AS is_false_positive,
        CASE WHEN s.model_score >= 600
              AND COALESCE(f.is_confirmed_fraud, 0) = 1
             THEN 1 ELSE 0 END             AS is_true_positive
    FROM fraud_db.transactions              t
    JOIN fraud_db.model_scores              s ON t.transaction_id = s.transaction_id
    LEFT JOIN fraud_db.confirmed_fraud_labels f ON t.transaction_id = f.transaction_id
    WHERE t.transaction_date BETWEEN :start_date AND :end_date
)

SELECT
    mcc,
    COUNT(*)                                          AS total_alerts,
    SUM(is_true_positive)                             AS true_positives,
    SUM(is_false_positive)                            AS false_positives,
    ROUND(SUM(is_false_positive)::FLOAT
          / NULLIF(COUNT(*), 0) * 100, 2)             AS fp_rate_pct,
    ROUND(SUM(is_true_positive)::FLOAT
          / NULLIF(SUM(is_true_positive)
                   + SUM(is_false_positive), 0) * 100, 2)  AS precision_pct,
    ROUND(SUM(CASE WHEN is_false_positive = 1
                   THEN amount ELSE 0 END), 2)        AS fp_amount_usd,
    ROUND(AVG(CASE WHEN is_false_positive = 1
                   THEN amount END), 2)               AS avg_fp_amount_usd,
    -- Estimated FP cost at $10/review
    SUM(is_false_positive) * 10                       AS estimated_fp_review_cost_usd
FROM alerts
GROUP BY mcc
ORDER BY false_positives DESC;


-- =============================================================================
-- PART 2 : FP by Customer Segment and Account Age
-- (New accounts have higher FP rate — expected, but quantify the impact)
-- =============================================================================

SELECT
    CASE
        WHEN a.account_age_days <  30  THEN '< 30 days'
        WHEN a.account_age_days <  90  THEN '30–90 days'
        WHEN a.account_age_days <  365 THEN '91–365 days'
        WHEN a.account_age_days <  730 THEN '1–2 years'
        ELSE                                '2+ years'
    END                                           AS account_age_bucket,
    a.customer_segment,
    COUNT(*)                                      AS total_alerts,
    SUM(al.is_true_positive)                      AS true_positives,
    SUM(al.is_false_positive)                     AS false_positives,
    ROUND(SUM(al.is_false_positive)::FLOAT
          / NULLIF(COUNT(*), 0) * 100, 2)         AS fp_rate_pct,
    ROUND(SUM(al.is_true_positive)::FLOAT
          / NULLIF(SUM(al.is_true_positive + al.is_false_positive), 0) * 100, 2) AS precision_pct
FROM alerts al
JOIN fraud_db.accounts a ON al.account_id = a.account_id
GROUP BY
    CASE
        WHEN a.account_age_days <  30  THEN '< 30 days'
        WHEN a.account_age_days <  90  THEN '30–90 days'
        WHEN a.account_age_days <  365 THEN '91–365 days'
        WHEN a.account_age_days <  730 THEN '1–2 years'
        ELSE                                '2+ years'
    END,
    a.customer_segment
ORDER BY fp_rate_pct DESC;


-- =============================================================================
-- PART 3 : Score Distribution of FPs (where is the threshold opportunity?)
-- Identify "safe" FPs that cluster just above threshold
-- =============================================================================

WITH fp_score_dist AS (
    SELECT
        FLOOR(s.model_score / 50) * 50                    AS score_bucket_lower,
        FLOOR(s.model_score / 50) * 50 + 49               AS score_bucket_upper,
        COUNT(*)                                           AS alert_count,
        SUM(CASE WHEN COALESCE(f.is_confirmed_fraud, 0) = 0 THEN 1 ELSE 0 END) AS fp_count,
        SUM(CASE WHEN COALESCE(f.is_confirmed_fraud, 0) = 1 THEN 1 ELSE 0 END) AS tp_count,
        SUM(CASE WHEN COALESCE(f.is_confirmed_fraud, 0) = 0
                 THEN t.amount ELSE 0 END)                 AS fp_amount_usd,
        SUM(CASE WHEN COALESCE(f.is_confirmed_fraud, 0) = 1
                 THEN t.amount ELSE 0 END)                 AS tp_amount_usd
    FROM fraud_db.model_scores              s
    JOIN fraud_db.transactions              t ON s.transaction_id = t.transaction_id
    LEFT JOIN fraud_db.confirmed_fraud_labels f ON s.transaction_id = f.transaction_id
    WHERE s.model_score >= 600  -- Only scored alerts
      AND t.transaction_date BETWEEN :start_date AND :end_date
    GROUP BY FLOOR(s.model_score / 50) * 50
)

SELECT
    CONCAT(score_bucket_lower, '–', score_bucket_upper) AS score_range,
    alert_count,
    fp_count,
    tp_count,
    ROUND(fp_count::FLOAT / NULLIF(alert_count, 0) * 100, 2)  AS fp_rate_pct,
    ROUND(fp_amount_usd, 2)                                     AS fp_amount_usd,
    ROUND(tp_amount_usd, 2)                                     AS tp_amount_usd,
    -- If we raised threshold to next bucket, what do we lose?
    SUM(tp_count) OVER (ORDER BY score_bucket_lower DESC)       AS cum_tp_from_top,
    SUM(fp_count) OVER (ORDER BY score_bucket_lower DESC)       AS cum_fp_from_top
FROM fp_score_dist
ORDER BY score_bucket_lower DESC;


-- =============================================================================
-- PART 4 : Threshold Sensitivity Analysis
-- Simulate detection rate and FP volume at different score cutoffs
-- Helps model owners and fraud operations agree on optimal operating point
-- =============================================================================

WITH thresholds AS (
    SELECT threshold_val
    FROM (VALUES (500),(550),(575),(600),(625),(650),(700),(750),(800)) t(threshold_val)
),

scored AS (
    SELECT
        t.transaction_id,
        t.amount,
        s.model_score,
        COALESCE(f.is_confirmed_fraud, 0) AS is_fraud
    FROM fraud_db.transactions              t
    JOIN fraud_db.model_scores              s ON t.transaction_id = s.transaction_id
    LEFT JOIN fraud_db.confirmed_fraud_labels f ON t.transaction_id = f.transaction_id
    WHERE t.transaction_date BETWEEN :start_date AND :end_date
)

SELECT
    th.threshold_val                                                    AS score_threshold,
    COUNT(sc.transaction_id)                                            AS total_transactions,
    SUM(sc.is_fraud)                                                    AS total_fraud,

    -- At this threshold
    SUM(CASE WHEN sc.model_score >= th.threshold_val
              AND sc.is_fraud = 1 THEN 1 ELSE 0 END)                   AS true_positives,
    SUM(CASE WHEN sc.model_score >= th.threshold_val
              AND sc.is_fraud = 0 THEN 1 ELSE 0 END)                   AS false_positives,
    SUM(CASE WHEN sc.model_score <  th.threshold_val
              AND sc.is_fraud = 1 THEN 1 ELSE 0 END)                   AS false_negatives,

    ROUND(SUM(CASE WHEN sc.model_score >= th.threshold_val
                    AND sc.is_fraud = 1 THEN 1 ELSE 0 END)::FLOAT
          / NULLIF(SUM(sc.is_fraud), 0) * 100, 2)                      AS detection_rate_pct,

    ROUND(SUM(CASE WHEN sc.model_score >= th.threshold_val
                    AND sc.is_fraud = 0 THEN 1 ELSE 0 END)::FLOAT
          / NULLIF(SUM(CASE WHEN sc.model_score >= th.threshold_val
                            THEN 1 ELSE 0 END), 0) * 100, 2)           AS fp_ratio_pct,

    -- Dollar detection at each threshold
    ROUND(SUM(CASE WHEN sc.model_score >= th.threshold_val
                    AND sc.is_fraud = 1 THEN sc.amount ELSE 0 END), 2) AS fraud_dollars_caught,
    ROUND(SUM(CASE WHEN sc.model_score <  th.threshold_val
                    AND sc.is_fraud = 1 THEN sc.amount ELSE 0 END), 2) AS fraud_dollars_escaped,

    -- Alert volume
    SUM(CASE WHEN sc.model_score >= th.threshold_val THEN 1 ELSE 0 END) AS alerts_generated,
    ROUND(SUM(CASE WHEN sc.model_score >= th.threshold_val
                   THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100, 3)      AS alert_rate_pct

FROM thresholds th
CROSS JOIN scored sc
GROUP BY th.threshold_val
ORDER BY th.threshold_val;
