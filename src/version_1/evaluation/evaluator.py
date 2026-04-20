# ROOT/src/version_1/evaluation/evaluator.py

from src.utils.data_configs import V1_TEST, V1_MODEL, V1_EVAL, V1_SHAP
from src.utils.logger import get_logger
from src.data_ingestion.load_data import load_data
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

import shap
import json

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────
TARGET_COL    = "isFraud"
V1_MODEL_PATH = V1_MODEL / "model.json"
V1_RESULT_DIR = V1_EVAL



# ── Functions ──────────────────────────────────────────────────
def load_model(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(path)
    logger.info(f"Model loaded from: {path}")
    return model


def separate_X_y(df: pd.DataFrame, target: str = TARGET_COL):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    logger.info(f"Features shape : {X.shape} | Target shape: {y.shape}")
    logger.info(f"Fraud rate     : {y.mean():.4f} ({y.mean() * 100:.2f}%)")
    return X, y


def get_predictions(model: xgb.XGBClassifier, X: pd.DataFrame):
    y_prob = model.predict_proba(X)[:, 1]
    logger.info(f"Predictions generated | Min: {y_prob.min():.4f} | Max: {y_prob.max():.4f} | Mean: {y_prob.mean():.4f}")
    return y_prob


def calculate_auc_roc(y_true, y_prob) -> float:
    auc_roc = roc_auc_score(y_true, y_prob)
    logger.info(f"AUC-ROC : {auc_roc:.4f}")
    return auc_roc


def calculate_auprc(y_true, y_prob) -> float:
    auprc = average_precision_score(y_true, y_prob)
    logger.info(f"AUPRC   : {auprc:.4f}")
    return auprc


def plot_pr_curve(y_true, y_prob, auprc: float, save_dir: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="crimson", lw=2,
             label=f"V1 XGBoost (AUPRC = {auprc:.4f})")
    plt.axhline(y=baseline, color="gray", linestyle="--", lw=1.5,
                label=f"Baseline (fraud rate = {baseline:.4f})")
    plt.xlabel("Recall",    fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("V1 XGBoost — Precision-Recall Curve (Test Set)", fontsize=13)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pr_curve_test.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"PR Curve saved to: {save_path}")


def log_final_results(auc_roc: float, auprc: float, save_dir: Path) -> None:
    lines = [
        "─── V1 XGBoost Evaluation Results (Test Set) ───────",
        f"  AUC-ROC  : {auc_roc:.4f}",
        f"  AUPRC    : {auprc:.4f}",
        f"  PR Curve : {save_dir / 'pr_curve_test.png'}",
        "────────────────────────────────────────────────────",
    ]
    for line in lines:
        logger.info(line)

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics.txt", "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Metrics saved to: {save_dir / 'metrics.txt'}")


def run_shap_analysis(model: xgb.XGBClassifier,
                      X_test: pd.DataFrame,
                      save_dir: Path,
                      top_n: int = 50) -> list:
    """
    Computes SHAP values on test set.
    Saves top N feature names to a JSON file for V2/V3 use.
    Returns list of top N feature names.
    """
    logger.info(f"Running SHAP analysis | top {top_n} features...")

    # ── Compute SHAP values ───────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ── Mean absolute SHAP per feature ────────────────────────
    # abs() because direction doesn't matter — magnitude does
    mean_shap = pd.Series(
        data  = abs(shap_values).mean(axis=0),
        index = X_test.columns
    ).sort_values(ascending=False)

    # ── Top N features ────────────────────────────────────────
    top_features = mean_shap.head(top_n).index.tolist()

    logger.info(f"Top {top_n} features identified")
    logger.info(f"Top 10 preview: {top_features[:10]}")

    # ── Save top features to JSON ─────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    features_path = save_dir / "top_features.json"

    with open(features_path, "w") as f:
        json.dump({"top_features": top_features}, f, indent=2)

    logger.info(f"Top {top_n} features saved to: {features_path}")

    # ── Save full SHAP importance to CSV ──────────────────────
    shap_path = save_dir / "shap_importance.csv"
    mean_shap.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    ).to_csv(shap_path, index=False)

    logger.info(f"Full SHAP importance saved to: {shap_path}")

    return top_features

# ── run() ──────────────────────────────────────────────────────
def run():
    logger.info("═══ V1 Evaluation Started ═══════════════════════════")

    # Step 1: Load model
    model = load_model(V1_MODEL_PATH)

    # Step 2: Load test data
    v1_test = load_data(V1_TEST)

    # Step 3: Separate X and y
    X_test, y_test = separate_X_y(v1_test)

    # Step 4: Predictions
    y_prob = get_predictions(model, X_test)

    # Step 5: AUC-ROC
    auc_roc = calculate_auc_roc(y_test, y_prob)

    # Step 6: AUPRC
    auprc = calculate_auprc(y_test, y_prob)

    # Step 7: PR Curve
    plot_pr_curve(y_test, y_prob, auprc, V1_RESULT_DIR)

    # Step 8: Log results
    log_final_results(auc_roc, auprc, V1_RESULT_DIR)

    run_shap_analysis(model, X_test, V1_SHAP, top_n=50)

    logger.info("═══ V1 Evaluation Completed ═════════════════════════")


if __name__ == "__main__":
    run()