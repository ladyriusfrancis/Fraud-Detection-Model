-- =============================================================================
-- 02_velocity_checks.sql
-- =============================================================================
-- Purpose : Real-time and batch velocity rule evaluation for fraud rule engine.
--           Velocity checks are applied BEFORE model scoring to enable fast,
--           rule-based decisioning for high-confidence fraud patterns.
--
--           Results feed into both:
--           (a) Real-time decisioning (Actimize AML/Fraud platform)
--           (b) Feature store for model scoring
--
-- Author  : Fraud Model Analytics
-- =============================================================================


-- -----------------------------------------------------------------------------
-- RULE V-01 : High transaction count in 1 hour (card testing / enumeration)
-- Threshold: > 5 transactions in 60 minutes triggers review
-- -----------------------------------------------------------------------------

SELECT
    account_id,
    COUNT(*)                                    AS txn_count_1h,
    SUM(amount)                                 AS total_amount_1h,
    MIN(amount)                                 AS min_txn_amount,
    MAX(amount)                                 AS max_txn_amount,
    COUNT(DISTINCT merchant_id)                 AS distinct_merchants_1h,
    'V-01: High Frequency - 1H'                AS rule_code,
    'HIGH'                                      AS risk_level
FROM fraud_db.transactions
WHERE transaction_timestamp >= DATEADD('hour', -1, CURRENT_TIMESTAMP)
  AND transaction_status IN ('SETTLED', 'PENDING')
GROUP BY account_id
HAVING COUNT(*) > 5
ORDER BY txn_count_1h DESC;


-- -----------------------------------------------------------------------------
-- RULE V-02 : Rapid amount escalation (probe → large)
-- Pattern: small txn (<$5) followed by large txn (>$500) within 2 hours
-- Classic card testing pattern
-- -----------------------------------------------------------------------------

WITH probe_txns AS (
    SELECT
        account_id,
        transaction_timestamp AS probe_time,
        amount                AS probe_amount,
        merchant_id           AS probe_merchant
    FROM fraud_db.transactions
    WHERE amount < 5.00
      AND transaction_timestamp >= DATEADD('day', -1, CURRENT_TIMESTAMP)
),

large_txns AS (
    SELECT
        account_id,
        transaction_id,
        transaction_timestamp AS large_time,
        amount                AS large_amount,
        merchant_id           AS large_merchant,
        channel
    FROM fraud_db.transactions
    WHERE amount > 500.00
      AND transaction_timestamp >= DATEADD('day', -1, CURRENT_TIMESTAMP)
)

SELECT
    l.transaction_id,
    l.account_id,
    p.probe_amount,
    p.probe_time,
    l.large_amount,
    l.large_time,
    DATEDIFF('minute', p.probe_time, l.large_time) AS minutes_between,
    l.channel,
    'V-02: Probe-to-Large Escalation'              AS rule_code,
    'HIGH'                                          AS risk_level
FROM large_txns l
JOIN probe_txns p
  ON l.account_id = p.account_id
 AND l.large_time BETWEEN p.probe_time
                      AND DATEADD('hour', 2, p.probe_time)
WHERE DATEDIFF('minute', p.probe_time, l.large_time) <= 120
ORDER BY l.large_time DESC;


-- -----------------------------------------------------------------------------
-- RULE V-03 : Multi-country velocity (impossible travel)
-- Pattern: transactions in > 1 country within 4 hours
-- -----------------------------------------------------------------------------

WITH country_txns AS (
    SELECT
        account_id,
        transaction_id,
        transaction_timestamp,
        amount,
        merchant_country,
        LEAD(merchant_country)       OVER (PARTITION BY account_id
                                           ORDER BY transaction_timestamp) AS next_country,
        LEAD(transaction_timestamp)  OVER (PARTITION BY account_id
                                           ORDER BY transaction_timestamp) AS next_txn_time,
        LEAD(amount)                 OVER (PARTITION BY account_id
                                           ORDER BY transaction_timestamp) AS next_amount
    FROM fraud_db.transactions
    WHERE transaction_date >= DATEADD('day', -7, CURRENT_DATE)
      AND transaction_status IN ('SETTLED', 'PENDING')
)

SELECT
    account_id,
    transaction_id,
    transaction_timestamp           AS first_txn_time,
    merchant_country                AS first_country,
    next_txn_time                   AS second_txn_time,
    next_country                    AS second_country,
    DATEDIFF('minute', transaction_timestamp, next_txn_time) AS minutes_apart,
    amount                          AS first_amount,
    next_amount                     AS second_amount,
    'V-03: Impossible Travel'       AS rule_code,
    'HIGH'                          AS risk_level
FROM country_txns
WHERE merchant_country <> next_country
  AND next_country IS NOT NULL
  AND DATEDIFF('hour', transaction_timestamp, next_txn_time) <= 4
ORDER BY transaction_timestamp DESC;


-- -----------------------------------------------------------------------------
-- RULE V-04 : Unusual spending spike vs. 30-day baseline
-- Triggers when 24h spend exceeds 300% of 30-day daily average
-- -----------------------------------------------------------------------------

WITH baseline AS (
    SELECT
        account_id,
        AVG(daily_spend)     AS avg_daily_spend_30d,
        STDDEV(daily_spend)  AS stddev_daily_spend_30d
    FROM (
        SELECT
            account_id,
            DATE_TRUNC('day', transaction_timestamp)::DATE AS txn_date,
            SUM(amount)                                     AS daily_spend
        FROM fraud_db.transactions
        WHERE transaction_date BETWEEN DATEADD('day', -31, CURRENT_DATE)
                                   AND DATEADD('day', -1, CURRENT_DATE)
          AND transaction_status = 'SETTLED'
        GROUP BY account_id, DATE_TRUNC('day', transaction_timestamp)::DATE
    ) daily
    GROUP BY account_id
),

recent_24h AS (
    SELECT
        account_id,
        SUM(amount)  AS spend_24h,
        COUNT(*)     AS txn_count_24h
    FROM fraud_db.transactions
    WHERE transaction_timestamp >= DATEADD('hour', -24, CURRENT_TIMESTAMP)
      AND transaction_status IN ('SETTLED', 'PENDING')
    GROUP BY account_id
)

SELECT
    r.account_id,
    r.spend_24h,
    r.txn_count_24h,
    b.avg_daily_spend_30d,
    b.stddev_daily_spend_30d,
    ROUND(r.spend_24h / NULLIF(b.avg_daily_spend_30d, 0), 2)  AS spend_ratio_vs_baseline,
    CASE
        WHEN r.spend_24h > b.avg_daily_spend_30d + 4 * b.stddev_daily_spend_30d THEN 'EXTREME'
        WHEN r.spend_24h > b.avg_daily_spend_30d + 3 * b.stddev_daily_spend_30d THEN 'HIGH'
        WHEN r.spend_24h > b.avg_daily_spend_30d + 2 * b.stddev_daily_spend_30d THEN 'MEDIUM'
    END                                                        AS severity,
    'V-04: Spend Spike vs Baseline'                            AS rule_code
FROM recent_24h r
JOIN baseline   b ON r.account_id = b.account_id
WHERE r.spend_24h > b.avg_daily_spend_30d * 3
  AND b.avg_daily_spend_30d > 0
ORDER BY spend_ratio_vs_baseline DESC;


-- -----------------------------------------------------------------------------
-- RULE V-05 : New device + high-value transaction (first use over $1,000)
-- Combines device intelligence signal with amount threshold
-- -----------------------------------------------------------------------------

SELECT
    t.transaction_id,
    t.account_id,
    t.amount,
    t.transaction_timestamp,
    t.channel,
    d.device_fingerprint_id,
    d.device_age_days,
    d.ip_risk_score,
    d.vpn_proxy_detected,
    d.geo_ip_mismatch,
    'V-05: New Device High Value'   AS rule_code,
    CASE
        WHEN d.ip_risk_score > 75 THEN 'HIGH'
        WHEN d.ip_risk_score > 50 THEN 'MEDIUM'
        ELSE 'LOW'
    END                             AS risk_level
FROM fraud_db.transactions      t
JOIN fraud_db.device_profiling  d
  ON t.transaction_id = d.transaction_id
WHERE t.amount > 1000
  AND d.device_age_days <= 3
  AND t.transaction_date >= DATEADD('day', -7, CURRENT_DATE)
  AND t.transaction_status IN ('SETTLED', 'PENDING')
ORDER BY t.amount DESC;


-- -----------------------------------------------------------------------------
-- RULE V-06 : Cross-account device sharing (device seen on multiple accounts)
-- Shared device fingerprints across accounts = mule network signal
-- -----------------------------------------------------------------------------

WITH device_account_counts AS (
    SELECT
        d.device_fingerprint_id,
        COUNT(DISTINCT t.account_id)    AS distinct_accounts,
        COUNT(DISTINCT t.transaction_id) AS txn_count,
        MIN(t.transaction_timestamp)    AS first_seen,
        MAX(t.transaction_timestamp)    AS last_seen,
        SUM(t.amount)                   AS total_amount
    FROM fraud_db.device_profiling  d
    JOIN fraud_db.transactions       t
      ON d.transaction_id = t.transaction_id
    WHERE t.transaction_date >= DATEADD('day', -30, CURRENT_DATE)
    GROUP BY d.device_fingerprint_id
    HAVING COUNT(DISTINCT t.account_id) >= 3  -- same device on 3+ distinct accounts
)

SELECT
    dac.device_fingerprint_id,
    dac.distinct_accounts,
    dac.txn_count,
    dac.first_seen,
    dac.last_seen,
    dac.total_amount,
    -- List of affected accounts for investigation
    LISTAGG(DISTINCT t.account_id, ', ')
        WITHIN GROUP (ORDER BY t.account_id)  AS affected_accounts,
    'V-06: Device Shared Across Accounts'     AS rule_code,
    'HIGH'                                    AS risk_level
FROM device_account_counts dac
JOIN fraud_db.device_profiling d ON dac.device_fingerprint_id = d.device_fingerprint_id
JOIN fraud_db.transactions     t ON d.transaction_id = t.transaction_id
WHERE t.transaction_date >= DATEADD('day', -30, CURRENT_DATE)
GROUP BY
    dac.device_fingerprint_id,
    dac.distinct_accounts,
    dac.txn_count,
    dac.first_seen,
    dac.last_seen,
    dac.total_amount
ORDER BY dac.distinct_accounts DESC, dac.total_amount DESC;
