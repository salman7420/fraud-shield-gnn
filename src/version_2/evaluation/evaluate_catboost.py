# src/version_2/evaluate_catboost.py
#
# Stage 4 — CatBoost Model Evaluation
# Loads trained model and val set, produces full evaluation suite.
#
# What this file does:
#   1. Loads trained model from result/model.cbm
#   2. Loads val set + cat_features
#   3. Generates predictions at multiple thresholds
#   4. Plots ROC curve
#   5. Plots Precision-Recall curve
#   6. Plots feature importance (top 30)
#   7. Plots confusion matrix
#   8. Saves all plots + full metrics report
#
# Usage (from project ROOT):
#   python -m src.version_2.evaluate_catboost

import sys
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ── Project root on path ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.data_configs import (
    V2_VAL,
    ARTIFACTS,
    V2_RESULT,
    V2_ALL_RESULT,
    V2_ALL_VAL
)

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────
ARTIFACTS_DIR     = ARTIFACTS
CAT_FEATURES_PATH = ARTIFACTS_DIR / "catboost_cat_features.json"
RESULT_DIR        = V2_RESULT
MODEL_PATH        = RESULT_DIR / "model.cbm"
METRICS_PATH      = RESULT_DIR / "metrics.json"
PLOTS_DIR         = RESULT_DIR / "plots"
EVAL_REPORT_PATH  = RESULT_DIR / "eval_report.json"

# ── Plot style constants ───────────────────────────────────
PLOT_STYLE   = "seaborn-v0_8-whitegrid"
FIG_DPI      = 150
FRAUD_COLOR  = "#e05c5c"
LEGIT_COLOR  = "#4c9be8"
MODEL_COLOR  = "#2a7f62"


# ══════════════════════════════════════════════════════════
#  STEP 1 — LOAD MODEL + DATA
# ══════════════════════════════════════════════════════════

def load_model_and_data() -> tuple:
    """
    Loads:
      - Trained CatBoost model from result/model.cbm
      - Val set from data/versions/v2_catboost/val.csv
      - cat_features from artifacts/catboost_cat_features.json

    Returns:
      model, val_pool, y_val, feature_names
    """
    # ── Model ──────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Run train_catboost.py first."
        )
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    logger.info(f"Model loaded: {MODEL_PATH}")

    # ── Cat features ───────────────────────────────────────
    if not CAT_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"cat_features not found: {CAT_FEATURES_PATH}\n"
            f"Run catboost_prep.py first."
        )
    with open(CAT_FEATURES_PATH, "r") as f:
        cat_features = json.load(f)["cat_features"]

    # ── Val data ───────────────────────────────────────────
    if not V2_ALL_VAL.exists():
        raise FileNotFoundError(
            f"Val split not found: {V2_VAL}\n"
            f"Run catboost_prep.py first."
        )
    val = pd.read_csv(V2_ALL_VAL)
    logger.info(
        f"Val set loaded: {val.shape[0]:,} rows × {val.shape[1]:,} cols | "
        f"fraud rate: {val['isFraud'].mean()*100:.2f}%"
    )

    feature_cols = [c for c in val.columns if c != "isFraud"]
    X_val = val[feature_cols]
    y_val = val["isFraud"].values

    available_cats = [c for c in cat_features if c in X_val.columns]

    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=available_cats,
    )

    return model, val_pool, y_val, feature_cols


# ══════════════════════════════════════════════════════════
#  STEP 2 — GENERATE PREDICTIONS
# ══════════════════════════════════════════════════════════

def generate_predictions(
    model:    CatBoostClassifier,
    val_pool: Pool,
    y_val:    np.ndarray,
) -> dict:
    """
    Generates fraud probability scores and evaluates at
    multiple thresholds to find the optimal F1 threshold.

    Why multiple thresholds:
      Default 0.5 is rarely optimal for imbalanced fraud data.
      At 3.5% fraud rate, a lower threshold (0.3-0.4) often
      gives better recall without destroying precision.

    Returns predictions dict with proba, best_threshold, metrics.
    """
    y_proba = model.predict_proba(val_pool)[:, 1]

    # ── Evaluate at multiple thresholds ───────────────────
    thresholds  = np.arange(0.1, 0.9, 0.05)
    threshold_results = []

    for thresh in thresholds:
        y_pred    = (y_proba >= thresh).astype(int)
        f1        = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall    = recall_score(y_val, y_pred, zero_division=0)
        threshold_results.append({
            "threshold": round(float(thresh), 2),
            "f1":        round(float(f1), 5),
            "precision": round(float(precision), 5),
            "recall":    round(float(recall), 5),
        })

    # ── Find threshold with best F1 ───────────────────────
    best = max(threshold_results, key=lambda x: x["f1"])
    best_threshold = best["threshold"]

    logger.info(
        f"Optimal threshold: {best_threshold} → "
        f"F1={best['f1']:.5f} | "
        f"Precision={best['precision']:.5f} | "
        f"Recall={best['recall']:.5f}"
    )

    # ── Final predictions at best threshold ───────────────
    y_pred_best = (y_proba >= best_threshold).astype(int)

    return {
        "y_proba":            y_proba,
        "y_pred":             y_pred_best,
        "best_threshold":     best_threshold,
        "threshold_results":  threshold_results,
        "best_metrics":       best,
    }


# ══════════════════════════════════════════════════════════
#  STEP 3 — COMPUTE FULL METRICS
# ══════════════════════════════════════════════════════════

def compute_full_metrics(
    y_val:   np.ndarray,
    preds:   dict,
) -> dict:
    """
    Computes the complete evaluation metrics suite:
      - AUC-ROC  (threshold-independent ranking)
      - AUPRC    (primary metric for class imbalance)
      - F1, Precision, Recall at best threshold
      - Confusion matrix breakdown
    """
    y_proba        = preds["y_proba"]
    y_pred         = preds["y_pred"]
    best_threshold = preds["best_threshold"]

    auc_roc   = roc_auc_score(y_val, y_proba)
    auprc     = average_precision_score(y_val, y_proba)
    f1        = f1_score(y_val, y_pred, zero_division=0)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall    = recall_score(y_val, y_pred, zero_division=0)
    cm        = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc":        round(float(auc_roc),   5),
        "auprc":          round(float(auprc),      5),
        "f1":             round(float(f1),         5),
        "precision":      round(float(precision),  5),
        "recall":         round(float(recall),     5),
        "threshold_used": best_threshold,
        "confusion_matrix": {
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
        },
        "threshold_sweep": preds["threshold_results"],
        "val_rows":        int(len(y_val)),
        "fraud_rate_val":  round(float(y_val.mean()), 5),
    }

    logger.info("─" * 50)
    logger.info(f"  AUC-ROC    : {auc_roc:.5f}")
    logger.info(f"  AUPRC      : {auprc:.5f}  ← primary metric")
    logger.info(f"  F1         : {f1:.5f}  (@ threshold={best_threshold})")
    logger.info(f"  Precision  : {precision:.5f}")
    logger.info(f"  Recall     : {recall:.5f}")
    logger.info(f"  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    logger.info("─" * 50)

    return metrics


# ══════════════════════════════════════════════════════════
#  STEP 4 — PLOTS
# ══════════════════════════════════════════════════════════

def plot_roc_curve(
    y_val:  np.ndarray,
    y_proba: np.ndarray,
) -> None:
    """Plots ROC curve with AUC score annotated."""
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc_score   = roc_auc_score(y_val, y_proba)

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color=MODEL_COLOR, lw=2,
            label=f"CatBoost v2  (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — CatBoost v2", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved: {path}")


def plot_pr_curve(
    y_val:   np.ndarray,
    y_proba: np.ndarray,
    best_threshold: float,
) -> None:
    """
    Plots Precision-Recall curve with AUPRC score.
    Marks the best F1 threshold point on the curve.
    Also shows the baseline (fraud rate = random classifier).
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(
        y_val, y_proba
    )
    auprc        = average_precision_score(y_val, y_proba)
    baseline     = y_val.mean()   # random classifier AUPRC = fraud rate

    # Find the point on the curve closest to best_threshold
    idx = np.argmin(np.abs(thresholds - best_threshold))

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall_vals, precision_vals,
            color=MODEL_COLOR, lw=2,
            label=f"CatBoost v2  (AUPRC = {auprc:.4f})")

    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Random baseline (AUPRC = {baseline:.4f})")

    # Mark best threshold point
    ax.scatter(recall_vals[idx], precision_vals[idx],
               color=FRAUD_COLOR, s=100, zorder=5,
               label=f"Best threshold = {best_threshold}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curve — CatBoost v2",
        fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = PLOTS_DIR / "pr_curve.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved: {path}")


def plot_feature_importance(model: CatBoostClassifier) -> None:
    """Horizontal bar chart of top 30 features by CatBoost gain importance."""
    fi = model.get_feature_importance(prettified=True)
    top30 = fi.head(30).sort_values("Importances", ascending=True)

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 10))

    bars = ax.barh(
        top30["Feature Id"],
        top30["Importances"],
        color=MODEL_COLOR,
        edgecolor="none",
        height=0.7,
    )

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center", ha="left", fontsize=8
        )

    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(
        "Top 30 Feature Importances — CatBoost v2",
        fontsize=14, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    path = PLOTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved: {path}")


def plot_confusion_matrix(
    y_val:  np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> None:
    """Plots confusion matrix as a clean heatmap with counts + percentages."""
    cm  = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_val)

    labels = [
        [f"TN\n{tn:,}\n({tn/total*100:.1f}%)",
         f"FP\n{fp:,}\n({fp/total*100:.1f}%)"],
        [f"FN\n{fn:,}\n({fn/total*100:.1f}%)",
         f"TP\n{tp:,}\n({tp/total*100:.1f}%)"],
    ]
    colors = np.array([[tn, fp], [fn, tp]], dtype=float)

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(colors, cmap="Blues", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"], fontsize=11)
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"], fontsize=11)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j],
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if colors[i, j] > colors.max() * 0.5
                    else "black")

    ax.set_title(
        f"Confusion Matrix — CatBoost v2  (threshold={threshold})",
        fontsize=13, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {path}")


def plot_threshold_sweep(preds: dict) -> None:
    """
    Line plot of F1, Precision, Recall across all thresholds.
    Helps visualise the tradeoff and confirm the best threshold choice.
    """
    results    = preds["threshold_results"]
    thresholds = [r["threshold"]  for r in results]
    f1s        = [r["f1"]         for r in results]
    precisions = [r["precision"]  for r in results]
    recalls    = [r["recall"]     for r in results]
    best_thresh = preds["best_threshold"]

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, f1s,        color=MODEL_COLOR,  lw=2, label="F1")
    ax.plot(thresholds, precisions, color=LEGIT_COLOR,  lw=2,
            linestyle="--", label="Precision")
    ax.plot(thresholds, recalls,    color=FRAUD_COLOR,  lw=2,
            linestyle=":",  label="Recall")
    ax.axvline(x=best_thresh, color="gray", linestyle="--", lw=1.5,
               label=f"Best threshold = {best_thresh}")

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Threshold Sweep — F1 / Precision / Recall",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_xlim([0.1, 0.85])
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    path = PLOTS_DIR / "threshold_sweep.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Threshold sweep plot saved: {path}")


# ══════════════════════════════════════════════════════════
#  STEP 5 — SAVE EVAL REPORT
# ══════════════════════════════════════════════════════════

def save_eval_report(metrics: dict) -> None:
    """
    Saves complete evaluation report to result/eval_report.json.
    Distinct from metrics.json (which is written during training).
    This file is the authoritative post-hoc evaluation output.
    """
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Eval report saved: {EVAL_REPORT_PATH}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def run_evaluation() -> dict:
    """Orchestrates all evaluation steps. Returns metrics dict."""
    start = time.time()

    logger.info("=" * 60)
    logger.info("CATBOOST EVALUATION — START")
    logger.info(f"Model   : {MODEL_PATH}")
    logger.info(f"Val set : {V2_ALL_VAL}")
    logger.info(f"Plots   : {PLOTS_DIR}")
    logger.info("=" * 60)

    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1 — Load
        logger.info("\n[ Step 1 ] Loading model and data ...")
        model, val_pool, y_val, feature_cols = load_model_and_data()

        # Step 2 — Predictions at multiple thresholds
        logger.info("\n[ Step 2 ] Generating predictions ...")
        preds = generate_predictions(model, val_pool, y_val)

        # Step 3 — Full metrics
        logger.info("\n[ Step 3 ] Computing metrics ...")
        metrics = compute_full_metrics(y_val, preds)

        # Step 4 — Plots
        logger.info("\n[ Step 4 ] Generating plots ...")
        plot_roc_curve(y_val, preds["y_proba"])
        plot_pr_curve(y_val, preds["y_proba"], preds["best_threshold"])
        plot_feature_importance(model)
        plot_confusion_matrix(y_val, preds["y_pred"], preds["best_threshold"])
        plot_threshold_sweep(preds)

        # Step 5 — Save report
        logger.info("\n[ Step 5 ] Saving eval report ...")
        save_eval_report(metrics)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"CATBOOST EVALUATION — COMPLETE ({elapsed:.1f}s)")
        logger.info("=" * 60)

        return metrics

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_evaluation()