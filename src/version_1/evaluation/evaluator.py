# ROOT/src/version_1/evaluation/evaluator.py

from src.utils.data_configs import V1_TEST, V1_MODEL, V1_EVAL
from src.utils.logger import get_logger
from src.data_ingestion.load_data import load_data

import pandas as pd
import xgboost as xgb
from pathlib import Path

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import os


logger = get_logger(__name__)

TARGET_COL    = "isFraud"
V1_MODEL_PATH = V1_MODEL   / "model.json"
V1_RESULT_DIR = V1_EVAL            # ROOT/src/version_1/results

# ── Step 1: Load saved model ───────────────────────────────────
def load_model(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(path)
    logger.info(f"Model loaded from: {path}")
    return model

model = load_model(V1_MODEL_PATH)

# ── Step 2: Load v1 test data ──────────────────────────────────
v1_test = load_data(V1_TEST)

# ── Step 3: Separate X and y ───────────────────────────────────
def separate_X_y(df: pd.DataFrame, target: str = TARGET_COL):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    logger.info(f"Features shape : {X.shape} | Target shape: {y.shape}")
    logger.info(f"Fraud rate     : {y.mean():.4f} ({y.mean() * 100:.2f}%)")
    return X, y

X_test, y_test = separate_X_y(v1_test)


# ── Step 4: Get predicted probabilities ───────────────────────
def get_predictions(model: xgb.XGBClassifier, X: pd.DataFrame):
    """
    Returns fraud probability score for each transaction.
    We use predict_proba not predict — we need scores not binary 0/1 labels.
    Column [:, 1] = probability of being fraud (class 1)
    """
    y_prob = model.predict_proba(X)[:, 1]
    logger.info(f"Predictions generated | Min: {y_prob.min():.4f} | Max: {y_prob.max():.4f} | Mean: {y_prob.mean():.4f}")
    return y_prob

y_prob = get_predictions(model, X_test)

# ── Step 5: Calculate AUC-ROC ─────────────────────────────────
def calculate_auc_roc(y_true, y_prob) -> float:
    auc_roc = roc_auc_score(y_true, y_prob)
    logger.info(f"AUC-ROC : {auc_roc:.4f}")
    return auc_roc

auc_roc = calculate_auc_roc(y_test, y_prob)

# ── Step 6: Calculate AUPRC ────────────────────────────────────
def calculate_auprc(y_true, y_prob) -> float:
    auprc = average_precision_score(y_true, y_prob)
    logger.info(f"AUPRC   : {auprc:.4f}")
    return auprc

auprc = calculate_auprc(y_test, y_prob)


# ── Step 7: Plot & save PR Curve ───────────────────────────────
def plot_pr_curve(y_true, y_prob, auprc: float, save_dir: Path) -> None:
    """
    Plots Precision-Recall curve and saves as PNG.
    Baseline = fraud rate (what a random classifier would score).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    plt.figure(figsize=(8, 6))

    # Model PR curve
    plt.plot(recall, precision,
             color="crimson", lw=2,
             label=f"V1 XGBoost (AUPRC = {auprc:.4f})")

    # Baseline — random classifier
    plt.axhline(y=baseline,
                color="gray", linestyle="--", lw=1.5,
                label=f"Baseline (fraud rate = {baseline:.4f})")

    plt.xlabel("Recall",    fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("V1 XGBoost — Precision-Recall Curve (Test Set)", fontsize=13)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pr_curve_test.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"PR Curve saved to: {save_path}")

plot_pr_curve(y_test, y_prob, auprc, V1_RESULT_DIR)

# ── Step 8: Log final results ──────────────────────────────────
def log_final_results(auc_roc: float, auprc: float, save_dir: Path) -> None:
    """
    Logs final metrics to console and saves them to a metrics.txt file.
    """
    lines = [
        "─── V1 XGBoost Evaluation Results (Test Set) ───────",
        f"  AUC-ROC  : {auc_roc:.4f}",
        f"  AUPRC    : {auprc:.4f}",
        f"  PR Curve : {save_dir / 'pr_curve_test.png'}",
        "────────────────────────────────────────────────────",
    ]

    # Log to console
    for line in lines:
        logger.info(line)

    # Save to file
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.txt"

    with open(metrics_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Metrics saved to: {metrics_path}")

log_final_results(auc_roc, auprc, V1_RESULT_DIR)