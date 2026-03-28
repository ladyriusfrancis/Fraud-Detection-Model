"""
generate_synthetic_data.py
==========================
Generates a realistic synthetic banking transaction dataset for fraud detection modeling.

Features include:
  - Core transaction attributes (amount, channel, merchant category, time-of-day)
  - Behavioral biometrics proxies (session duration, login typing speed, navigation pattern score)
  - Device profiling attributes (device fingerprint, OS, IP reputation score, VPN flag)
  - Account-level risk signals (account age, prior disputes, velocity in last 24h)
  - LexisNexis-style risk score simulation

Usage:
    python generate_synthetic_data.py
    --> Outputs: transactions.csv (100,000 rows)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import hashlib

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_TRANSACTIONS = 100_000
FRAUD_RATE = 0.025  # 2.5% — realistic for banking card fraud


def generate_account_pool(n_accounts: int = 8000) -> pd.DataFrame:
    """Create a pool of synthetic customer accounts with risk attributes."""
    account_ids = [f"ACC{str(i).zfill(6)}" for i in range(n_accounts)]
    account_ages_days = np.random.exponential(scale=730, size=n_accounts).astype(int)  # avg 2 years
    account_ages_days = np.clip(account_ages_days, 1, 5475)

    prior_disputes = np.random.poisson(lam=0.3, size=n_accounts)
    credit_limit = np.random.choice([1000, 2500, 5000, 10000, 15000, 25000], size=n_accounts,
                                    p=[0.10, 0.20, 0.30, 0.25, 0.10, 0.05])

    return pd.DataFrame({
        "account_id": account_ids,
        "account_age_days": account_ages_days,
        "prior_disputes": prior_disputes,
        "credit_limit": credit_limit,
    })


def generate_device_fingerprint() -> str:
    """Generate a pseudo-random device fingerprint hash."""
    raw = f"{random.randint(0, 10_000_000)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def simulate_transactions(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate transaction records with fraud labels.
    Fraud transactions exhibit statistically different patterns:
      - Higher amounts relative to account credit limit
      - Unusual hours (late night / early morning)
      - New or flagged devices
      - High IP risk scores
      - Short session durations (bot-like behavior)
      - Card-not-present channels dominate
    """
    records = []
    txn_timestamps = [
        datetime(2024, 1, 1) + timedelta(seconds=random.randint(0, 365 * 24 * 3600))
        for _ in range(N_TRANSACTIONS)
    ]
    txn_timestamps.sort()

    is_fraud_arr = np.random.binomial(1, FRAUD_RATE, size=N_TRANSACTIONS)

    merchant_categories = [
        "retail", "grocery", "restaurant", "gas_station", "travel",
        "entertainment", "electronics", "online_marketplace", "atm", "wire_transfer"
    ]
    # Fraud concentrates in high-risk merchant categories
    fraud_mcc_weights = [0.05, 0.02, 0.03, 0.03, 0.10, 0.05, 0.20, 0.30, 0.10, 0.12]
    legit_mcc_weights = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]

    channels = ["in_store", "online", "mobile_app", "atm", "phone"]
    fraud_channel_weights = [0.05, 0.50, 0.20, 0.15, 0.10]
    legit_channel_weights = [0.40, 0.25, 0.20, 0.10, 0.05]

    device_pool_legit = [generate_device_fingerprint() for _ in range(500)]
    device_pool_fraud = [generate_device_fingerprint() for _ in range(3000)]  # many unique devices = fraud signal

    account_sample = accounts.sample(N_TRANSACTIONS, replace=True).reset_index(drop=True)

    for i in range(N_TRANSACTIONS):
        is_fraud = int(is_fraud_arr[i])
        ts = txn_timestamps[i]
        acct = account_sample.iloc[i]
        hour = ts.hour

        # --- Amount ---
        if is_fraud:
            # Fraudsters often probe with small amounts then escalate, or go near credit limit
            fraud_amount_type = np.random.choice(["probe", "large"], p=[0.35, 0.65])
            if fraud_amount_type == "probe":
                amount = round(np.random.uniform(0.50, 5.00), 2)
            else:
                amount = round(np.random.uniform(acct["credit_limit"] * 0.5, acct["credit_limit"] * 1.1), 2)
        else:
            amount = round(np.random.lognormal(mean=4.2, sigma=1.1), 2)
            amount = min(amount, acct["credit_limit"] * 0.8)

        # --- Merchant Category ---
        mcc = np.random.choice(merchant_categories,
                               p=(fraud_mcc_weights if is_fraud else legit_mcc_weights))

        # --- Channel ---
        channel = np.random.choice(channels,
                                   p=(fraud_channel_weights if is_fraud else legit_channel_weights))

        # --- Time features ---
        is_weekend = 1 if ts.weekday() >= 5 else 0
        is_night = 1 if (hour >= 23 or hour <= 4) else 0

        # Fraud skews heavily toward night hours
        if is_fraud:
            hour = np.random.choice(range(24), p=[
                0.08, 0.09, 0.08, 0.07, 0.06, 0.03,  # 0-5 AM
                0.02, 0.01, 0.01, 0.01, 0.01, 0.01,  # 6-11 AM
                0.02, 0.02, 0.02, 0.02, 0.03, 0.04,  # noon-5 PM
                0.06, 0.07, 0.08, 0.08, 0.07, 0.08   # 6 PM - 11 PM
            ])
            is_night = 1 if (hour >= 23 or hour <= 4) else 0

        # --- Behavioral Biometrics (proxy features) ---
        # These would come from a behavioral biometrics platform (e.g., BioCatch)
        if is_fraud:
            # Bots/mules: very fast or very erratic typing, short sessions
            session_duration_sec = int(np.random.exponential(scale=45))   # very short sessions
            typing_speed_wpm = round(np.random.uniform(5, 25), 1)         # abnormal typing speed
            mouse_linearity_score = round(np.random.uniform(0.1, 0.5), 3) # robotic or erratic mouse movement
            copy_paste_detected = np.random.binomial(1, 0.70)             # credential stuffing signals
            login_attempt_count = np.random.poisson(lam=3.5)              # multiple login attempts
        else:
            session_duration_sec = int(np.random.lognormal(mean=5.5, sigma=1.0))
            typing_speed_wpm = round(np.random.normal(loc=55, scale=12), 1)
            mouse_linearity_score = round(np.random.normal(loc=0.78, scale=0.10), 3)
            copy_paste_detected = np.random.binomial(1, 0.08)
            login_attempt_count = np.random.poisson(lam=1.1)

        session_duration_sec = max(5, session_duration_sec)
        typing_speed_wpm = max(1, typing_speed_wpm)
        mouse_linearity_score = np.clip(mouse_linearity_score, 0.0, 1.0)

        # --- Device Profiling ---
        # Simulates data from device intelligence platforms (e.g., ThreatMetrix / LexisNexis)
        if is_fraud:
            device_fingerprint = random.choice(device_pool_fraud)
            ip_risk_score = round(np.random.beta(a=5, b=2) * 100, 1)      # high IP risk
            vpn_proxy_detected = np.random.binomial(1, 0.55)
            device_age_days = int(np.random.exponential(scale=10))         # new/unknown devices
            os_type = np.random.choice(["Windows", "Android", "iOS", "Linux", "Unknown"],
                                       p=[0.30, 0.25, 0.15, 0.10, 0.20])
            known_device = 0
        else:
            device_fingerprint = random.choice(device_pool_legit)
            ip_risk_score = round(np.random.beta(a=2, b=8) * 100, 1)      # low IP risk
            vpn_proxy_detected = np.random.binomial(1, 0.05)
            device_age_days = int(np.random.exponential(scale=300))
            os_type = np.random.choice(["Windows", "Android", "iOS", "macOS", "Linux"],
                                       p=[0.30, 0.25, 0.28, 0.12, 0.05])
            known_device = 1 if device_age_days > 30 else 0

        # --- LexisNexis-style composite risk score (simulated) ---
        # In production this would be pulled via LexisNexis Risk Solutions API
        if is_fraud:
            lexisnexis_risk_score = int(np.random.beta(a=6, b=2) * 1000)
        else:
            lexisnexis_risk_score = int(np.random.beta(a=2, b=7) * 1000)

        # --- Velocity signals (simplified — full velocity done in SQL) ---
        txn_count_24h = max(1, int(np.random.poisson(lam=(8 if is_fraud else 2.5))))
        distinct_merchants_24h = min(txn_count_24h, max(1, int(np.random.poisson(lam=(5 if is_fraud else 1.8)))))
        amount_sum_24h = round(amount * txn_count_24h * np.random.uniform(0.7, 1.3), 2)

        # --- Geographic mismatch ---
        geo_mismatch = np.random.binomial(1, 0.60 if is_fraud else 0.04)

        records.append({
            "transaction_id": f"TXN{str(i).zfill(8)}",
            "account_id": acct["account_id"],
            "transaction_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_date": ts.strftime("%Y-%m-%d"),
            "transaction_hour": hour,
            "is_weekend": is_weekend,
            "is_night_transaction": is_night,
            "amount": amount,
            "credit_limit": acct["credit_limit"],
            "amount_to_limit_ratio": round(amount / acct["credit_limit"], 4),
            "merchant_category": mcc,
            "channel": channel,
            "is_card_not_present": 1 if channel in ["online", "mobile_app", "phone"] else 0,
            # Behavioral biometrics
            "session_duration_sec": session_duration_sec,
            "typing_speed_wpm": typing_speed_wpm,
            "mouse_linearity_score": mouse_linearity_score,
            "copy_paste_detected": copy_paste_detected,
            "login_attempt_count": login_attempt_count,
            # Device profiling
            "device_fingerprint": device_fingerprint,
            "device_age_days": device_age_days,
            "known_device": known_device,
            "os_type": os_type,
            "ip_risk_score": ip_risk_score,
            "vpn_proxy_detected": vpn_proxy_detected,
            # Third-party risk score
            "lexisnexis_risk_score": lexisnexis_risk_score,
            # Velocity
            "txn_count_24h": txn_count_24h,
            "distinct_merchants_24h": distinct_merchants_24h,
            "amount_sum_24h": amount_sum_24h,
            # Account attributes
            "account_age_days": acct["account_age_days"],
            "prior_disputes": acct["prior_disputes"],
            "geo_mismatch": geo_mismatch,
            # Label
            "is_fraud": is_fraud,
        })

    return pd.DataFrame(records)


def main():
    print("Generating account pool...")
    accounts = generate_account_pool(n_accounts=8000)

    print(f"Generating {N_TRANSACTIONS:,} transactions ({FRAUD_RATE*100:.1f}% fraud rate)...")
    df = simulate_transactions(accounts)

    out_path = "transactions.csv"
    df.to_csv(out_path, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"\n✓ Dataset saved to: {out_path}")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Fraudulent         : {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    print(f"  Legitimate         : {len(df)-fraud_count:,}")
    print(f"  Features           : {df.shape[1]-1}")
    print(f"\nColumn summary:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
