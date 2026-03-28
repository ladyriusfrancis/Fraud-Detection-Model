-- =============================================================================
-- 01_feature_extraction.sql
-- =============================================================================
-- Purpose : Extract and engineer transaction-level features for fraud scoring.
--           Designed for use against the core banking transaction warehouse.
--           Compatible with: Snowflake, Redshift, SQL Server (minor dialect tweaks).
--
-- Author  : Fraud Model Analytics
-- Updated : 2024-Q4
-- =============================================================================


-- -----------------------------------------------------------------------------
-- SECTION 1 : Base transaction + account context
-- -----------------------------------------------------------------------------

WITH base_txns AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.transaction_timestamp,
        t.amount,
        t.merchant_category_code                          AS mcc,
        t.channel,
        t.is_card_not_present,
        t.merchant_id,
        t.merchant_country,
        -- Account snapshot at time of transaction
        a.account_open_date,
        DATEDIFF('day', a.account_open_date,
                 t.transaction_timestamp)                 AS account_age_days,
        a.credit_limit,
        ROUND(t.amount / NULLIF(a.credit_limit, 0), 4)  AS amount_to_limit_ratio,
        a.customer_segment,                               -- e.g., mass, mass_affluent, hni
        a.prior_dispute_count
    FROM fraud_db.transactions           t
    JOIN fraud_db.accounts               a
      ON t.account_id = a.account_id
    WHERE t.transaction_date BETWEEN :start_date AND :end_date
      AND t.transaction_status = 'SETTLED'
),


-- -----------------------------------------------------------------------------
-- SECTION 2 : Time-based features
-- -----------------------------------------------------------------------------

time_features AS (
    SELECT
        transaction_id,
        EXTRACT(HOUR   FROM transaction_timestamp)      AS txn_hour,
        EXTRACT(DOW    FROM transaction_timestamp)      AS txn_day_of_week,  -- 0=Sun
        CASE WHEN EXTRACT(DOW FROM transaction_timestamp)
                  IN (0, 6) THEN 1 ELSE 0 END          AS is_weekend,
        CASE WHEN EXTRACT(HOUR FROM transaction_timestamp)
                  BETWEEN 23 AND 24
                  OR EXTRACT(HOUR FROM transaction_timestamp)
                  BETWEEN 0  AND 4  THEN 1 ELSE 0 END  AS is_night_txn,
        CASE WHEN EXTRACT(MONTH FROM transaction_timestamp)
                  IN (11, 12, 1) THEN 1 ELSE 0 END     AS is_holiday_season
    FROM base_txns
),


-- -----------------------------------------------------------------------------
-- SECTION 3 : Velocity features (rolling windows on account)
-- Velocity = # / $ of transactions in a time window prior to current transaction
-- High velocity is a strong fraud signal
-- -----------------------------------------------------------------------------

velocity_features AS (
    SELECT
        t.transaction_id,

        -- 1-hour velocity
        COUNT(h.transaction_id)           AS txn_count_1h,
        COALESCE(SUM(h.amount), 0)        AS amount_sum_1h,
        COUNT(DISTINCT h.merchant_id)     AS distinct_merchants_1h,

        -- 6-hour velocity
        COUNT(s.transaction_id)           AS txn_count_6h,
        COALESCE(SUM(s.amount), 0)        AS amount_sum_6h,
        COUNT(DISTINCT s.merchant_id)     AS distinct_merchants_6h,

        -- 24-hour velocity
        COUNT(d.transaction_id)           AS txn_count_24h,
        COALESCE(SUM(d.amount), 0)        AS amount_sum_24h,
        COUNT(DISTINCT d.merchant_id)     AS distinct_merchants_24h,
        COUNT(DISTINCT d.merchant_country) AS distinct_countries_24h,

        -- 7-day velocity (baseline behavior reference)
        COUNT(w.transaction_id)           AS txn_count_7d,
        COALESCE(SUM(w.amount), 0)        AS amount_sum_7d,
        COALESCE(AVG(w.amount), 0)        AS avg_txn_amount_7d,
        COALESCE(STDDEV(w.amount), 0)     AS stddev_txn_amount_7d

    FROM base_txns t

    -- Prior 1 hour
    LEFT JOIN fraud_db.transactions h
           ON h.account_id = t.account_id
          AND h.transaction_timestamp >= DATEADD('hour', -1, t.transaction_timestamp)
          AND h.transaction_timestamp <  t.transaction_timestamp
          AND h.transaction_status = 'SETTLED'

    -- Prior 6 hours
    LEFT JOIN fraud_db.transactions s
           ON s.account_id = t.account_id
          AND s.transaction_timestamp >= DATEADD('hour', -6, t.transaction_timestamp)
          AND s.transaction_timestamp <  t.transaction_timestamp
          AND s.transaction_status = 'SETTLED'

    -- Prior 24 hours
    LEFT JOIN fraud_db.transactions d
           ON d.account_id = t.account_id
          AND d.transaction_timestamp >= DATEADD('hour', -24, t.transaction_timestamp)
          AND d.transaction_timestamp <  t.transaction_timestamp
          AND d.transaction_status = 'SETTLED'

    -- Prior 7 days
    LEFT JOIN fraud_db.transactions w
           ON w.account_id = t.account_id
          AND w.transaction_timestamp >= DATEADD('day', -7, t.transaction_timestamp)
          AND w.transaction_timestamp <  t.transaction_timestamp
          AND w.transaction_status = 'SETTLED'

    GROUP BY t.transaction_id
),


-- -----------------------------------------------------------------------------
-- SECTION 4 : Behavioral biometrics features
-- Source: BioCatch or equivalent behavioral biometrics platform API feed
-- Joined on session_id linked to transaction_id in session mapping table
-- -----------------------------------------------------------------------------

biometric_features AS (
    SELECT
        sm.transaction_id,
        bb.session_duration_seconds,
        bb.typing_speed_wpm,
        bb.typing_rhythm_consistency_score,  -- 0–1; low = bot-like
        bb.mouse_linearity_score,            -- 0–1; very high or very low = suspicious
        bb.copy_paste_event_count,
        bb.login_attempt_count,
        bb.idle_time_ratio,                  -- proportion of session idle
        bb.hesitation_score,                 -- pauses inconsistent with account holder behavior
        CASE WHEN bb.login_attempt_count > 2
             THEN 1 ELSE 0 END               AS multiple_login_attempts,
        CASE WHEN bb.copy_paste_event_count > 0
              AND bb.session_duration_seconds < 60
             THEN 1 ELSE 0 END               AS credential_stuffing_signal
    FROM fraud_db.session_txn_mapping sm
    JOIN biometrics_db.session_features bb
      ON sm.session_id = bb.session_id
),


-- -----------------------------------------------------------------------------
-- SECTION 5 : Device profiling features
-- Source: LexisNexis ThreatMetrix / device intelligence platform
-- -----------------------------------------------------------------------------

device_features AS (
    SELECT
        dp.transaction_id,
        dp.device_fingerprint_id,
        dp.device_type,                       -- mobile, desktop, tablet
        dp.os_type,
        dp.browser_type,
        dp.device_age_days,                   -- days since device first seen
        CASE WHEN dp.device_age_days < 7
             THEN 1 ELSE 0 END                AS is_new_device,
        dp.ip_risk_score,                     -- 0–100; LexisNexis composite score
        dp.vpn_proxy_detected,
        dp.tor_exit_node_detected,
        dp.ip_country_code,
        CASE WHEN dp.ip_country_code <>
                  acct_country.country_code
             THEN 1 ELSE 0 END                AS geo_ip_mismatch,
        dp.device_blacklisted,                -- device appeared in prior confirmed fraud
        dp.true_ip_address                    -- resolved through proxy detection
    FROM fraud_db.device_profiling dp
    JOIN fraud_db.account_country  acct_country
      ON dp.account_id = acct_country.account_id
),


-- -----------------------------------------------------------------------------
-- SECTION 6 : LexisNexis Risk Solutions composite score
-- Pulled daily via batch API; scored at transaction time
-- -----------------------------------------------------------------------------

lexisnexis_scores AS (
    SELECT
        lx.transaction_id,
        lx.composite_risk_score,             -- 0–1000; higher = riskier
        lx.identity_risk_indicator,          -- flags synthetic identity signals
        lx.network_risk_indicator,           -- known fraud ring associations
        lx.velocity_risk_indicator,          -- cross-institution velocity flag
        CASE WHEN lx.composite_risk_score >= 700 THEN 'HIGH'
             WHEN lx.composite_risk_score >= 400 THEN 'MEDIUM'
             ELSE 'LOW'
        END                                  AS ln_risk_band
    FROM lexisnexis_db.risk_scores lx
    WHERE lx.score_date = CURRENT_DATE
),


-- -----------------------------------------------------------------------------
-- SECTION 7 : Amount deviation from account's historical baseline
-- Z-score of current transaction vs. account's rolling 30-day distribution
-- -----------------------------------------------------------------------------

amount_deviation AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.amount                                                    AS current_amount,
        hist.avg_amount_30d,
        hist.stddev_amount_30d,
        CASE WHEN hist.stddev_amount_30d = 0 THEN 0
             ELSE (t.amount - hist.avg_amount_30d)
                  / hist.stddev_amount_30d
        END                                                         AS amount_zscore_30d,
        CASE WHEN t.amount > hist.avg_amount_30d + 3 * hist.stddev_amount_30d
             THEN 1 ELSE 0 END                                      AS amount_outlier_flag
    FROM base_txns t
    JOIN (
        SELECT
            account_id,
            AVG(amount)    AS avg_amount_30d,
            STDDEV(amount) AS stddev_amount_30d
        FROM fraud_db.transactions
        WHERE transaction_date BETWEEN DATEADD('day', -30, :start_date)
                                   AND :start_date
          AND transaction_status = 'SETTLED'
        GROUP BY account_id
    ) hist ON t.account_id = hist.account_id
),


-- -----------------------------------------------------------------------------
-- SECTION 8 : Final feature set assembly
-- -----------------------------------------------------------------------------

final_features AS (
    SELECT
        -- Identifiers
        b.transaction_id,
        b.account_id,
        b.transaction_timestamp,

        -- Core transaction
        b.amount,
        b.mcc,
        b.channel,
        b.is_card_not_present,
        b.account_age_days,
        b.credit_limit,
        b.amount_to_limit_ratio,
        b.prior_dispute_count,

        -- Time features
        tf.txn_hour,
        tf.txn_day_of_week,
        tf.is_weekend,
        tf.is_night_txn,
        tf.is_holiday_season,

        -- Velocity
        vf.txn_count_1h,
        vf.amount_sum_1h,
        vf.txn_count_6h,
        vf.amount_sum_6h,
        vf.txn_count_24h,
        vf.amount_sum_24h,
        vf.distinct_merchants_24h,
        vf.distinct_countries_24h,
        vf.avg_txn_amount_7d,
        vf.stddev_txn_amount_7d,

        -- Amount deviation
        ad.amount_zscore_30d,
        ad.amount_outlier_flag,

        -- Behavioral biometrics
        bf.session_duration_seconds,
        bf.typing_speed_wpm,
        bf.mouse_linearity_score,
        bf.copy_paste_event_count,
        bf.login_attempt_count,
        bf.multiple_login_attempts,
        bf.credential_stuffing_signal,

        -- Device profiling
        df.device_age_days,
        df.is_new_device,
        df.ip_risk_score,
        df.vpn_proxy_detected,
        df.tor_exit_node_detected,
        df.geo_ip_mismatch,
        df.device_blacklisted,

        -- LexisNexis
        ls.composite_risk_score          AS ln_composite_score,
        ls.identity_risk_indicator       AS ln_identity_risk,
        ls.network_risk_indicator        AS ln_network_risk,
        ls.velocity_risk_indicator       AS ln_velocity_risk,
        ls.ln_risk_band

    FROM base_txns              b
    LEFT JOIN time_features     tf ON b.transaction_id = tf.transaction_id
    LEFT JOIN velocity_features vf ON b.transaction_id = vf.transaction_id
    LEFT JOIN amount_deviation  ad ON b.transaction_id = ad.transaction_id
    LEFT JOIN biometric_features bf ON b.transaction_id = bf.transaction_id
    LEFT JOIN device_features   df ON b.transaction_id = df.transaction_id
    LEFT JOIN lexisnexis_scores ls ON b.transaction_id = ls.transaction_id
)

SELECT * FROM final_features
ORDER BY transaction_timestamp;
