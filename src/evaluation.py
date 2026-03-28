"""
evaluation.py
=============
Comprehensive model evaluation and KPI reporting for fraud detection models.

Produces:
  - Precision-Recall and ROC curves
  - Score distribution plots (legit vs. fraud)
  - Threshold sensitivity analysis (detection rate vs. FPR trade-off)
  - SHAP feature importance plots (global + waterfall for individual cases)
  - Dollar-weighted fraud detection analysis
  - Confusion matrix visualization
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "fraud":   "#E63946",
    "legit":   "#457B9D",
    "neutral": "#6C757D",
    "accent":  "#F4A261",
}


# ---------------------------------------------------------------------------
# 1. ROC and Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    y_test: pd.Series,
    models: dict[str, Any],
    X_test: pd.DataFrame,
    save: bool = True,
) -> None:
    """Plot ROC and PR curves for multiple models side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance: ROC & Precision-Recall Curves",
                 fontsize=13, fontweight="bold", y=1.01)

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr_arr, tpr_arr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        axes[1].plot(rec_arr, prec_arr, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    # ROC axis
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Detection Rate)", fontsize=11)
    axes[0].set_title("ROC Curve", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # PR axis
    fraud_rate = y_test.mean()
    axes[1].axhline(y=fraud_rate, color="k", linestyle="--", lw=1,
                    label=f"No-skill baseline ({fraud_rate*100:.1f}%)")
    axes[1].set_xlabel("Recall (Detection Rate)", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Precision-Recall Curve", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURE_DIR / 'roc_pr_curves.png'}")
    plt.show()


# ---------------------------------------------------------------------------
# 2. Score distribution (separation of fraud vs. legit)
# ---------------------------------------------------------------------------

def plot_score_distribution(
    y_test: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "XGBoost",
    threshold: float = 0.5,
    save: bool = True,
) -> None:
    """
    Visualize model score distribution for fraud vs. legitimate transactions.
    Good separation indicates strong discriminating power.
    """
    fraud_scores = y_prob[y_test == 1]
    legit_scores = y_prob[y_test == 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)

    ax.hist(legit_scores, bins=bins, alpha=0.6, color=PALETTE["legit"],
            label=f"Legitimate (n={len(legit_scores):,})", density=True)
    ax.hist(fraud_scores, bins=bins, alpha=0.7, color=PALETTE["fraud"],
            label=f"Fraud (n={len(fraud_scores):,})", density=True)

    ax.axvline(threshold, color=PALETTE["accent"], lw=2.5, linestyle="--",
               label=f"Decision threshold = {threshold}")

    ax.set_xlabel("Model Score (Fraud Probability)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{model_name} — Score Distribution: Fraud vs. Legitimate",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate separation
    ks_stat = abs(
        np.searchsorted(np.sort(legit_scores), threshold) / len(legit_scores)
        - np.searchsorted(np.sort(fraud_scores), threshold) / len(fraud_scores)
    )
    ax.text(0.98, 0.95, f"Threshold KS ≈ {ks_stat:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=PALETTE["neutral"])

    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURE_DIR / 'score_distribution.png'}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Threshold sensitivity analysis
# ---------------------------------------------------------------------------

def threshold_sensitivity_table(
    y_test: pd.Series,
    y_prob: np.ndarray,
    amounts: pd.Series | None = None,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compute KPIs across a range of decision thresholds.
    Helps fraud operations and model owners agree on the optimal cutoff.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    rows = []
    total_fraud = y_test.sum()

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())

        detection_rate = tp / max(total_fraud, 1)
        fpr            = fp / max(fp + tn, 1)
        precision      = tp / max(tp + fp, 1)
        alerts         = tp + fp

        row: dict[str, Any] = {
            "threshold":      t,
            "detection_rate": round(detection_rate * 100, 2),
            "fpr_pct":        round(fpr * 100, 4),
            "precision_pct":  round(precision * 100, 2),
            "alerts":         alerts,
            "tp":             tp,
            "fp":             fp,
            "fn":             fn,
            "fp_per_tp":      round(fp / max(tp, 1), 1),
        }

        if amounts is not None:
            fraud_dollars   = amounts[y_test == 1].sum()
            caught_dollars  = amounts[(y_pred == 1) & (y_test == 1)].sum()
            escaped_dollars = amounts[(y_pred == 0) & (y_test == 1)].sum()
            row["fraud_exposure_usd"]   = round(fraud_dollars, 2)
            row["fraud_caught_usd"]     = round(caught_dollars, 2)
            row["fraud_escaped_usd"]    = round(escaped_dollars, 2)
            row["dollar_detection_pct"] = round(caught_dollars / max(fraud_dollars, 1) * 100, 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n--- Threshold Sensitivity Analysis ---")
    print(df.to_string(index=False))
    return df


def plot_threshold_sensitivity(sensitivity_df: pd.DataFrame, save: bool = True) -> None:
    """Visualize detection rate vs. FPR and alert volume across thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Threshold Sensitivity Analysis", fontsize=13,
                 fontweight="bold", y=1.01)

    x = sensitivity_df["threshold"]

    # Detection rate and FPR
    ax1 = axes[0]
    ax1.plot(x, sensitivity_df["detection_rate"], "o-",
             color=PALETTE["fraud"], lw=2, label="Detection Rate (%)")
    ax1.set_ylabel("Detection Rate (%)", color=PALETTE["fraud"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=PALETTE["fraud"])

    ax1b = ax1.twinx()
    ax1b.plot(x, sensitivity_df["fpr_pct"], "s--",
              color=PALETTE["legit"], lw=2, label="False Positive Rate (%)")
    ax1b.set_ylabel("False Positive Rate (%)", color=PALETTE["legit"], fontsize=11)
    ax1b.tick_params(axis="y", labelcolor=PALETTE["legit"])

    ax1.set_xlabel("Decision Threshold", fontsize=11)
    ax1.set_title("Detection Rate vs. False Positive Rate", fontsize=11)
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # Alert volume
    axes[1].bar(x.astype(str), sensitivity_df["tp"], color=PALETTE["fraud"],
                label="True Positives", alpha=0.85)
    axes[1].bar(x.astype(str), sensitivity_df["fp"], bottom=sensitivity_df["tp"],
                color=PALETTE["legit"], label="False Positives", alpha=0.85)
    axes[1].set_xlabel("Decision Threshold", fontsize=11)
    axes[1].set_ylabel("Alert Count", fontsize=11)
    axes[1].set_title("Alert Composition by Threshold", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "threshold_sensitivity.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURE_DIR / 'threshold_sensitivity.png'}")
    plt.show()


# ---------------------------------------------------------------------------
# 4. SHAP feature importance
# ---------------------------------------------------------------------------

def plot_shap_importance(
    model: Any,
    X_test: pd.DataFrame,
    max_display: int = 20,
    save: bool = True,
) -> shap.Explanation:
    """
    Generate SHAP summary plot for model explainability.
    Critical for communicating model decisions to fraud operations and audit.
    """
    print("Computing SHAP values (this may take ~1 minute)...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer(X_sample)

    # Summary beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance — XGBoost Fraud Model",
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURE_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURE_DIR / 'shap_summary.png'}")
    plt.show()

    # Bar plot (mean |SHAP|)
    fig2, _ = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("Mean |SHAP| Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        fig2.savefig(FIGURE_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURE_DIR / 'shap_bar.png'}")
    plt.show()

    return shap_values


# ---------------------------------------------------------------------------
# 5. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """Visualize confusion matrix with dollar-intuitive labeling."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Legitimate", "Fraud"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=12, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save:
        fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(FIGURE_DIR / fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURE_DIR / fname}")
    plt.show()
