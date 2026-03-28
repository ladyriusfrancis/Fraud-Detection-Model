"""
feature_engineering.py
=======================
Transforms raw transaction data into model-ready features.

Feature groups:
  1. Transaction-level features  (amount ratios, channel encoding)
  2. Time features               (cyclical encoding, night/weekend flags)
  3. Velocity features           (rolling window aggregations)
  4. Behavioral biometrics       (session and typing signals)
  5. Device profiling            (device age, IP risk, VPN flags)
  6. Amount deviation            (z-score vs. account baseline)
  7. Interaction features        (high-value × new-device, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


# ---------------------------------------------------------------------------
# 1. Time features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode temporal signals including cyclical hour encoding."""
    df = df.copy()

    hour = df["transaction_hour"].astype(float)

    # Cyclical encoding: hour of day (avoids 23→0 discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Binary risk flags
    df["is_night_txn"]    = ((hour >= 23) | (hour <= 4)).astype(int)
    df["is_business_hrs"] = ((hour >= 9) & (hour <= 17)).astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. Amount deviation (z-score vs. account-level 7-day baseline)
# ---------------------------------------------------------------------------

def add_amount_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how much each transaction deviates from the account's recent baseline.
    A high z-score indicates an unusually large (or small) transaction.
    """
    df = df.copy()

    grp = df.groupby("account_id")["amount"]
    df["acct_avg_amount_7d"]    = grp.transform("mean")
    df["acct_stddev_amount_7d"] = grp.transform("std").fillna(1.0)

    df["amount_zscore"] = (
        (df["amount"] - df["acct_avg_amount_7d"])
        / df["acct_stddev_amount_7d"].replace(0, 1)
    ).round(4)

    df["amount_outlier_flag"] = (df["amount_zscore"].abs() > 3).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Velocity ratio features
# ---------------------------------------------------------------------------

def add_velocity_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ratios between velocity windows to capture acceleration patterns.
    A sudden spike in 1h/24h ratio signals potential fraud.
    """
    df = df.copy()

    # Txn count acceleration
    df["velocity_accel_1h_24h"] = (
        df["txn_count_1h"] / df["txn_count_24h"].replace(0, 1)
    ).round(4)

    # Amount acceleration
    df["amount_accel_1h_24h"] = (
        df["amount_sum_1h"] / df["amount_sum_24h"].replace(0, 1)
    ).round(4)

    # Merchant diversity ratio
    df["merchant_diversity_ratio"] = (
        df["distinct_merchants_24h"] / df["txn_count_24h"].replace(0, 1)
    ).round(4)

    return df


# ---------------------------------------------------------------------------
# 4. Behavioral biometric composite signals
# ---------------------------------------------------------------------------

def add_biometric_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive composite risk signals from behavioral biometrics.
    These proxy for bot activity, credential stuffing, and account takeover.
    """
    df = df.copy()

    # Typing speed anomaly: very fast (<15 wpm) or very slow (>100 wpm) = suspicious
    df["typing_speed_anomaly"] = (
        (df["typing_speed_wpm"] < 15) | (df["typing_speed_wpm"] > 100)
    ).astype(int)

    # Short session + copy-paste = strong bot / credential stuffing signal
    df["bot_signal"] = (
        (df["session_duration_sec"] < 30) & (df["copy_paste_detected"] == 1)
    ).astype(int)

    # Login stress: multiple attempts in a short session
    df["login_stress_score"] = (
        df["login_attempt_count"] * (1 / df["session_duration_sec"].clip(lower=1))
    ).round(6)

    return df


# ---------------------------------------------------------------------------
# 5. Device risk composite features
# ---------------------------------------------------------------------------

def add_device_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine device profiling signals into composite risk indicators.
    Mirrors logic applied in LexisNexis ThreatMetrix integration.
    """
    df = df.copy()

    # New device + high IP risk = strong takeover signal
    df["new_device_high_ip_risk"] = (
        (df["device_age_days"] < 7) & (df["ip_risk_score"] > 60)
    ).astype(int)

    # VPN or proxy on a card-not-present transaction
    df["cnp_with_anonymizer"] = (
        (df["vpn_proxy_detected"] == 1) & (df["is_card_not_present"] == 1)
    ).astype(int)

    # Normalized IP risk (scale to 0-1)
    df["ip_risk_score_norm"] = (df["ip_risk_score"] / 100.0).round(4)

    # LexisNexis score normalized
    df["ln_risk_score_norm"] = (df["lexisnexis_risk_score"] / 1000.0).round(4)

    return df


# ---------------------------------------------------------------------------
# 6. Categorical encoding
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = ["merchant_category", "channel", "os_type"]

def encode_categoricals(
    df: pd.DataFrame,
    encoder: OrdinalEncoder | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Ordinal-encode low-cardinality categorical columns.
    Returns (transformed_df, fitted_encoder).
    """
    df = df.copy()

    # Fill unseen categories
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("unknown")

    if fit:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[CATEGORICAL_COLS] = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    return df, encoder


# ---------------------------------------------------------------------------
# 7. Master pipeline: apply all transformations
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Core transaction
    "amount",
    "amount_to_limit_ratio",
    "is_card_not_present",
    "account_age_days",
    "prior_disputes",
    "credit_limit",

    # Time
    "hour_sin",
    "hour_cos",
    "is_night_txn",
    "is_business_hrs",
    "is_weekend",

    # Amount deviation
    "amount_zscore",
    "amount_outlier_flag",

    # Velocity (raw)
    "txn_count_1h",
    "amount_sum_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "distinct_merchants_24h",

    # Velocity ratios
    "velocity_accel_1h_24h",
    "amount_accel_1h_24h",
    "merchant_diversity_ratio",

    # Behavioral biometrics
    "session_duration_sec",
    "typing_speed_wpm",
    "mouse_linearity_score",
    "copy_paste_detected",
    "login_attempt_count",
    "typing_speed_anomaly",
    "bot_signal",
    "login_stress_score",

    # Device profiling
    "device_age_days",
    "known_device",
    "ip_risk_score_norm",
    "vpn_proxy_detected",
    "new_device_high_ip_risk",
    "cnp_with_anonymizer",

    # LexisNexis
    "ln_risk_score_norm",
    "geo_mismatch",

    # Categoricals (encoded)
    "merchant_category",
    "channel",
    "os_type",
]


def build_features(
    df: pd.DataFrame,
    encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = True,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Apply the full feature engineering pipeline.

    Parameters
    ----------
    df          : Raw transaction DataFrame
    encoder     : Pre-fitted OrdinalEncoder (pass during inference)
    fit_encoder : Whether to fit a new encoder (True for training)

    Returns
    -------
    X           : Feature DataFrame (columns = FEATURE_COLUMNS)
    encoder     : Fitted OrdinalEncoder
    """
    df = add_time_features(df)
    df = add_amount_deviation_features(df)
    df = add_velocity_ratio_features(df)
    df = add_biometric_composite_features(df)
    df = add_device_risk_features(df)
    df, encoder = encode_categoricals(df, encoder=encoder, fit=fit_encoder)

    # Select and order feature columns
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    X = X.fillna(0)

    return X, encoder


# ---------------------------------------------------------------------------
# Sklearn-compatible transformer wrapper
# ---------------------------------------------------------------------------

class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Drop-in sklearn transformer for use in Pipeline objects.
    """

    def __init__(self):
        self.encoder_: OrdinalEncoder | None = None

    def fit(self, X: pd.DataFrame, y=None):
        _, self.encoder_ = build_features(X, fit_encoder=True)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        features, _ = build_features(X, encoder=self.encoder_, fit_encoder=False)
        return features

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_COLUMNS)
