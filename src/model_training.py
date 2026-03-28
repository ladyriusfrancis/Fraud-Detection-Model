"""
model_training.py
=================
Trains, tunes, and saves fraud detection models.

Models included:
  - LogisticRegression    : Interpretable baseline
  - XGBClassifier         : Primary production model
  - LGBMClassifier        : Alternative gradient boosting (often faster)

Handles class imbalance via:
  - scale_pos_weight (XGBoost / LGBM)
  - class_weight='balanced' (Logistic Regression)
  - Optuna-based hyperparameter optimization

Usage:
    python model_training.py --data-path data/transactions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from feature_engineering import build_features, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load transaction CSV and return (raw_df, labels)."""
    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    y  = df["is_fraud"].astype(int)
    return df, y


def split_train_test(
    df: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-aware train/test split.
    Sorts by transaction_timestamp so test set is always 'future' data,
    mimicking real deployment conditions.
    """
    df = df.sort_values("transaction_timestamp").reset_index(drop=True)
    y  = y.loc[df.index].reset_index(drop=True)

    cutoff = int(len(df) * (1 - test_ratio))
    return (
        df.iloc[:cutoff].reset_index(drop=True),
        df.iloc[cutoff:].reset_index(drop=True),
        y.iloc[:cutoff].reset_index(drop=True),
        y.iloc[cutoff:].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Baseline model: Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Regularized logistic regression baseline.
    Provides interpretable coefficients for stakeholder communication.
    """
    log.info("Training Logistic Regression baseline...")
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    log.info("  Logistic Regression training complete.")
    return model


# ---------------------------------------------------------------------------
# Primary model: XGBoost with Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def _xgb_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
) -> float:
    """Optuna objective for XGBoost — maximizes PR-AUC (better for imbalanced data)."""
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 800),
        "max_depth":          trial.suggest_int("max_depth", 3, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":   scale_pos_weight,
        "eval_metric":        "aucpr",
        "random_state":       RANDOM_STATE,
        "n_jobs":             -1,
        "verbosity":          0,
    }
    model = XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )
    return scores.mean()


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> XGBClassifier:
    """
    Tune and train XGBoost with Optuna.
    Uses PR-AUC as the optimization target (appropriate for imbalanced fraud data).
    """
    neg_count  = (y_train == 0).sum()
    pos_count  = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    log.info(f"Class balance — Legitimate: {neg_count:,} | Fraud: {pos_count:,} "
             f"| scale_pos_weight: {scale_pos_weight:.1f}")

    log.info(f"Running Optuna hyperparameter search ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["eval_metric"]      = "aucpr"
    best_params["random_state"]     = RANDOM_STATE
    best_params["n_jobs"]           = -1
    best_params["verbosity"]        = 0

    log.info(f"Best PR-AUC: {study.best_value:.4f}")
    log.info(f"Best params: {json.dumps(best_params, indent=2)}")

    log.info("Training final XGBoost model on full training set...")
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    # Persist best params for documentation
    params_path = MODEL_DIR / "xgb_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute fraud-specific model KPIs:
      - ROC-AUC, PR-AUC
      - Detection Rate (Recall) at decision threshold
      - False Positive Rate at decision threshold
      - Precision at decision threshold
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    detection_rate = tp / max(tp + fn, 1)   # Recall
    fpr            = fp / max(fp + tn, 1)
    precision      = tp / max(tp + fp, 1)
    f1             = (2 * precision * detection_rate / max(precision + detection_rate, 1e-9))

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    metrics = {
        "model":          model_name,
        "threshold":      threshold,
        "roc_auc":        round(roc_auc, 4),
        "pr_auc":         round(pr_auc, 4),
        "detection_rate": round(detection_rate, 4),
        "precision":      round(precision, 4),
        "fpr":            round(fpr, 6),
        "f1_score":       round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }

    log.info(f"\n{'='*55}")
    log.info(f"  {model_name} — Evaluation @ threshold={threshold}")
    log.info(f"{'='*55}")
    log.info(f"  ROC-AUC        : {roc_auc:.4f}")
    log.info(f"  PR-AUC         : {pr_auc:.4f}")
    log.info(f"  Detection Rate : {detection_rate*100:.2f}%")
    log.info(f"  Precision      : {precision*100:.2f}%")
    log.info(f"  False Pos Rate : {fpr*100:.4f}%")
    log.info(f"  F1 Score       : {f1:.4f}")
    log.info(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")

    return metrics


def save_model(model: Any, name: str) -> Path:
    """Serialize model to disk using pickle."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EverBank fraud detection models")
    parser.add_argument("--data-path",  default="data/transactions.csv")
    parser.add_argument("--n-trials",   type=int, default=50,
                        help="Optuna trials for XGBoost tuning")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Decision threshold for binary classification metrics")
    args = parser.parse_args()

    # 1. Load data
    raw_df, y = load_data(args.data_path)

    # 2. Time-aware split
    train_df, test_df, y_train, y_test = split_train_test(raw_df, y)
    log.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # 3. Feature engineering
    log.info("Building features...")
    X_train, encoder = build_features(train_df, fit_encoder=True)
    X_test,  _       = build_features(test_df,  encoder=encoder, fit_encoder=False)
    log.info(f"Feature matrix shape: {X_train.shape}")

    # 4. Save encoder
    save_model(encoder, "ordinal_encoder")

    # 5. Train models
    lr_model  = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train, n_trials=args.n_trials)

    # 6. Evaluate
    all_metrics = []
    for name, model in [("LogisticRegression", lr_model), ("XGBoost", xgb_model)]:
        metrics = evaluate_model(model, X_test, y_test,
                                 model_name=name,
                                 threshold=args.threshold)
        all_metrics.append(metrics)
        save_model(model, name.lower())

    # 7. Save metrics summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    log.info(f"\nMetrics summary saved to {MODEL_DIR / 'model_comparison.csv'}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
